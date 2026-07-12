"""Vendor-template extractor.

When the v3 ParsedDocument's layout fingerprint matches a vendor template
stored in proc.bp_extraction_template (managed by extraction_v2's
PostgresTemplateStore), emit each stored field hint as a Candidate with
confidence 0.9. These are deterministic high-confidence locators learned
from prior extractions / human corrections.

Substring guarantee: hint.value must be present as a substring of
parsed.full_text. Hints whose value isn't in full_text are dropped (this
can happen if the template was learned on a different format of the same
vendor; we don't risk hallucination).
"""
from __future__ import annotations
import hashlib
import re
import logging
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema
from src.services.extraction_v3.yaml_schema.registry import register_extractor
from src.services.extraction_v2.template_store_pg import PostgresTemplateStore
from src.services.db import get_conn

log = logging.getLogger(__name__)

_LABEL_HEAD_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z\s/&\-]{1,40}?)\s*([:#]|$)"
)


def _label_set(text: str) -> set[str]:
    labels = set()
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line or len(line) > 80:
            continue
        m = _LABEL_HEAD_RE.match(line)
        if not m:
            continue
        token = m.group(1).strip().lower()
        if len(token) < 3:
            continue
        labels.add(token)
    return labels


def _fmt_v3_to_v2(file_format: str) -> str:
    if file_format in ("pdf-native", "pdf-scanned"):
        return "pdf"
    return file_format


def compute_v3_fingerprint(parsed: ParsedDocument) -> str:
    """Compute the layout fingerprint of a v3 ParsedDocument using the same
    algorithm as v2 so the stored templates are compatible."""
    parts: list[str] = []
    parts.append(f"fmt={_fmt_v3_to_v2(parsed.file_format)}")
    parts.append(f"pages={len(parsed.pages)}")
    labels = sorted(_label_set(parsed.full_text))
    parts.append("labels=" + "|".join(labels))
    table_col_counts = []
    n_tables = 0
    for page in parsed.pages:
        for table in page.tables:
            n_tables += 1
            if table.rows:
                ncols = max((len(row) for row in table.rows), default=0)
                table_col_counts.append(ncols)
    parts.append("tables=" + str(n_tables))
    parts.append("table_cols=" + "|".join(str(c) for c in sorted(table_col_counts)))
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _bbox_for_value(parsed: ParsedDocument, value: str):
    val_l = value.lower().strip()
    for page in parsed.pages:
        for tok in page.tokens:
            if val_l in tok.text.lower():
                return (page.index, tok.bbox)
    return (0, (0.0, 0.0, 0.0, 0.0))


@register_extractor("vendor_template")
class VendorTemplateExtractor(Extractor):

    def __init__(self):
        self._store = PostgresTemplateStore(get_conn)

    def produce_candidates(self, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        active = [f.name for f in schema.fields if "vendor_template" in f.extractors]
        if not active:
            return []
        try:
            fp = compute_v3_fingerprint(parsed)
        except Exception:
            log.debug("fingerprint compute failed", exc_info=True)
            return []
        try:
            template = self._store.get(fp)
        except Exception:
            log.debug("template store lookup failed", exc_info=True)
            return []
        if template is None:
            return []
        candidates = []
        for fname, hint in template.field_hints.items():
            if fname not in active:
                continue
            value = (hint.value or "").strip()
            if not value or value not in parsed.full_text:
                continue  # substring guarantee
            page_idx, b = _bbox_for_value(parsed, value)
            confidence = float(hint.confidence) if hint.confidence is not None else 0.9
            candidates.append(Candidate(
                field=fname,
                value=value,
                page=page_idx,
                bbox=b,
                evidence_text=value,
                model="vendor_template",
                confidence=min(0.95, confidence),  # cap below 1.0
            ))
        return candidates
