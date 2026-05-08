"""High-level template service used by the production orchestrator.

The orchestrator calls into this module at two points per extraction:

  1. AFTER an extraction (structural OR legacy LLM) produces a header +
     line_items, call ``apply_template`` to override the LLM's output for
     fields where a stored template has a more reliable hint. This is the
     accuracy lever for known vendors — it deterministically replaces
     supplier_name hallucinations etc.

  2. After persistence, if the extraction quality is good, call
     ``learn_from_extraction`` to snapshot the values as a candidate
     template. Successive extractions of the same fingerprint then
     benefit from the snapshot.

The store is a singleton — at import time the service tries to wire a
``PostgresTemplateStore`` via the supplied connection factory; if that
fails (e.g., during tests with no DB) it falls back to an in-memory
store so import never crashes.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from src.services.extraction_v2.fingerprint import compute_fingerprint
from src.services.extraction_v2.template_store import (
    FieldHint, InMemoryTemplateStore, LineItemHints, TemplateStore,
    VendorTemplate,
)
from src.services.structural_extractor.parsing.model import ParsedDocument

logger = logging.getLogger(__name__)


__all__ = [
    "TemplateService", "configure_template_service", "get_template_service",
]


# Fields where a template hint should override the per-doc extraction.
# These are the fields where LLM hallucination is the primary failure
# mode in production (see I-18 in the issues report).
_HEADER_OVERRIDE_FIELDS = {
    "supplier_name", "supplier_id", "buyer_name", "buyer_id",
}


class TemplateService:
    """Singleton-style facade over a TemplateStore.

    Tests inject an InMemoryTemplateStore; production wires a Postgres-backed
    store via :func:`configure_template_service`.
    """

    def __init__(self, store: TemplateStore):
        self.store = store

    # -- fingerprint helpers ------------------------------------------------

    def fingerprint(self, doc: ParsedDocument) -> str:
        return compute_fingerprint(doc)

    # -- read path: applied at extraction time ------------------------------

    def apply_template(
        self, header: dict[str, Any], fingerprint: str,
        *, line_items: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Override fields in `header` with template hints, where present.

        Returns ``(updated_header, source_map)`` where ``source_map`` is a
        ``{field_name: source}`` dict recording which fields were
        template-overridden vs. left as-is. The orchestrator uses the
        source map to populate the provenance sidecar table.
        """
        tpl = self.store.get(fingerprint)
        if tpl is None:
            return header, {}

        result = dict(header)
        source_map: dict[str, str] = {}
        for field_name, hint in tpl.field_hints.items():
            if field_name not in _HEADER_OVERRIDE_FIELDS:
                continue
            current = result.get(field_name)
            template_value = hint.value
            if template_value is None:
                continue
            # Only override when the LLM/structural value is empty OR
            # disagrees with the template AND the template confidence is
            # high. We deliberately do NOT override high-confidence
            # values that DO match — that would just re-write what we
            # already have.
            if current and str(current).strip() == str(template_value).strip():
                continue
            result[field_name] = template_value
            source_map[field_name] = "template"
            logger.info(
                "[template] %s: applied template for fingerprint=%s — %s='%s' (was '%s')",
                tpl.vendor_name or "unknown", fingerprint[:8], field_name,
                template_value, current,
            )
        if source_map:
            self.store.record_success(
                fingerprint, fields_committed=tuple(source_map.keys())
            )
        return result, source_map

    # -- prompt-augmentation helpers --------------------------------------

    def line_items_prompt_hint(self, fingerprint: str) -> Optional[str]:
        """Return a vendor-specific prompt suffix to attach to the
        line-items salvage prompt for documents matching ``fingerprint``.

        Used by the orchestrator's `_llm_salvage_line_items` to bias the
        LLM toward the column layout we have learned for this vendor.
        Returns ``None`` if the template is unknown or has no line-item
        hints, in which case the salvage falls back to the generic
        prompt.
        """
        tpl = self.store.get(fingerprint)
        if tpl is None or tpl.line_item_hints is None:
            return None
        hints = tpl.line_item_hints
        bits: list[str] = []
        if tpl.vendor_name:
            bits.append(f"This document is from {tpl.vendor_name}.")
        if hints.column_map:
            cols_text = ", ".join(sorted(hints.column_map.values()))
            bits.append(
                f"Previous successful extractions of this layout used these "
                f"columns: {cols_text}. Map the table headers to the matching "
                f"column names."
            )
        if hints.header_anchors:
            anchors = ", ".join(repr(a) for a in hints.header_anchors[:6])
            bits.append(f"Look for the line-items table near: {anchors}.")
        if hints.expected_min_rows >= 1:
            bits.append(
                f"This vendor's documents typically contain at least "
                f"{hints.expected_min_rows} line item(s) — search the "
                f"entire document body before returning an empty list."
            )
        return " ".join(bits) if bits else None

    def legacy_extraction_prompt_context(
        self, fingerprint: str,
    ) -> Optional[str]:
        """Return a prompt-injectable summary of EVERY known fact for this
        vendor — vendor name, learned supplier_name/buyer_name, line-item
        column hints — for use by the legacy LLM's MAIN extraction call.

        This is the prompt-augmentation lever for vendors we have not yet
        rewritten to V2. Strict rule: never produce a context string that
        embeds raw values likely to mis-direct the LLM (no IDs, no
        amounts) — only stable layout/identity hints.
        """
        tpl = self.store.get(fingerprint)
        if tpl is None:
            return None
        bits: list[str] = []
        if tpl.vendor_name:
            bits.append(f"This document is from vendor: {tpl.vendor_name}.")
        for field_name in ("supplier_name", "buyer_name"):
            hint = tpl.field_hints.get(field_name)
            if hint and hint.value:
                bits.append(
                    f"Confirmed canonical {field_name} for this vendor: "
                    f"{hint.value!r}. Prefer this value when extracting "
                    f"{field_name}."
                )
        line_hint = self.line_items_prompt_hint(fingerprint)
        if line_hint:
            bits.append(line_hint)
        return "\n".join(bits) if bits else None

    # -- write path: applied after extraction succeeds ---------------------

    def learn_from_extraction(
        self,
        fingerprint: str,
        header: dict[str, Any],
        line_items: list[dict[str, Any]],
        *,
        doc_type: str,
        vendor_name_hint: Optional[str] = None,
        rescued_fields: Optional[set[str]] = None,
    ) -> bool:
        """Snapshot the header as a candidate template.

        We learn aggressively but conservatively:
          - Only fields that look "trustworthy" are recorded — non-empty,
            non-garbage strings, plus any field that was rescued from the
            filename (the filename is ground truth in this codebase).
          - We never overwrite an existing template's hints — if a human
            already onboarded this vendor we treat their values as
            authoritative.
          - Returns True if a template was created or extended.
        """
        if not header:
            return False
        existing = self.store.get(fingerprint)
        # If a human-onboarded template exists (correction_count > 0) we
        # leave it alone — their corrections beat auto-learned snapshots.
        if existing is not None and existing.correction_count > 0:
            return False

        rescued_fields = rescued_fields or set()
        new_hints: dict[str, FieldHint] = (
            dict(existing.field_hints) if existing else {}
        )
        learned = False
        for field_name in _HEADER_OVERRIDE_FIELDS:
            value = header.get(field_name)
            if not value or not isinstance(value, str):
                continue
            value = value.strip()
            if len(value) < 2:
                continue
            # Do not learn from values that look auto-generated or
            # garbage-shaped (e.g. URLs, "INVOICE NUMBER:" fragments).
            lowered = value.lower()
            if any(marker in lowered for marker in (
                ".com", ".co.uk", "://", "invoice number",
                "purchase order", "@",
            )):
                continue
            # Confidence: 0.99 for filename-rescued fields (filename is
            # canonical in this repo), 0.90 for LLM-derived but agreeing
            # with what's currently in the header.
            conf = 0.99 if field_name in rescued_fields else 0.90
            existing_hint = new_hints.get(field_name)
            if existing_hint and existing_hint.value == value:
                continue
            new_hints[field_name] = FieldHint(
                field=field_name, value=value, confidence=conf,
                label=None, anchor=None,
            )
            learned = True

        # Snapshot / refine a line-item layout summary whenever the
        # extraction returned rows. This is the auto-learning loop for
        # line-items column layouts:
        #   - First success: write a fresh hint based on observed columns.
        #   - Subsequent success: refine ``column_map`` (union of columns)
        #     and bump ``expected_min_rows`` to max(existing, observed)
        #     when the row count is materially higher than what's stored.
        # We do NOT shrink ``expected_min_rows``; once we've seen a doc
        # with N lines we keep that floor for future invariant checks.
        line_hints = existing.line_item_hints if existing else None
        if line_items:
            columns_seen: set[str] = set()
            for item in line_items:
                for col in item.keys():
                    if isinstance(col, str):
                        columns_seen.add(col)
            new_columns = {
                c: c for c in sorted(columns_seen)
                if isinstance(c, str) and c.isidentifier()
            }
            observed_min_rows = max(1, len(line_items))
            if line_hints is None:
                line_hints = LineItemHints(
                    header_anchors=[],
                    column_map=new_columns,
                    expected_min_rows=observed_min_rows,
                )
                learned = True
            else:
                merged_columns = dict(line_hints.column_map)
                merged_columns.update(new_columns)
                merged_min_rows = max(
                    int(line_hints.expected_min_rows or 1),
                    observed_min_rows,
                )
                if (merged_columns != line_hints.column_map
                        or merged_min_rows != line_hints.expected_min_rows):
                    line_hints = LineItemHints(
                        header_anchors=list(line_hints.header_anchors),
                        column_map=merged_columns,
                        expected_min_rows=merged_min_rows,
                    )
                    learned = True

        if not learned:
            return False

        tpl = VendorTemplate(
            fingerprint=fingerprint,
            vendor_name=(existing.vendor_name if existing else vendor_name_hint),
            doc_type=doc_type,
            field_hints=new_hints,
            line_item_hints=line_hints,
            success_count=(existing.success_count if existing else 0),
            correction_count=(existing.correction_count if existing else 0),
        )
        self.store.upsert(tpl)
        logger.info(
            "[template] auto-learned template for fingerprint=%s vendor=%s "
            "(%d hints, %d line cols)",
            fingerprint[:8], tpl.vendor_name or "unknown",
            len(new_hints),
            len(line_hints.column_map) if line_hints else 0,
        )
        return True


# -- module-level singleton ------------------------------------------------

_service: Optional[TemplateService] = None
_lock = threading.Lock()


def configure_template_service(get_conn: Optional[Callable] = None) -> TemplateService:
    """Configure (or replace) the module-level template service.

    Called once during app startup with a Postgres connection factory.
    Tests pass ``get_conn=None`` to force the in-memory backend.
    """
    global _service
    with _lock:
        if get_conn is None:
            _service = TemplateService(InMemoryTemplateStore())
        else:
            try:
                from src.services.extraction_v2.template_store_pg import (
                    PostgresTemplateStore,
                )
                _service = TemplateService(PostgresTemplateStore(get_conn))
            except Exception:
                logger.exception(
                    "[template] PostgresTemplateStore init failed — "
                    "falling back to in-memory store"
                )
                _service = TemplateService(InMemoryTemplateStore())
        return _service


def get_template_service() -> TemplateService:
    """Lazy-init in-memory service if no production config has run."""
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = TemplateService(InMemoryTemplateStore())
    return _service
