"""Single-flow extraction dispatch.

Orchestrates L0 (parse) → L1 (regex) → L3 grounding + invariants → persist.
L2 engineered modules (table, NER, address, date, bbox proximity) and the
LLM judge are wired into the gaps that this MVP path leaves open — added
incrementally as live testing exposes them.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Any
from uuid import uuid4

from src.services.extraction import persistence, promotion
from src.services.extraction.engineered.ner_validator import fill_ner_gaps
from src.services.extraction.engineered.table_extractor import extract_line_items
from src.services.extraction.judge_gate import run_grounded_judge_for_gaps
from src.services.extraction.parser import parse as parse_document
from src.services.extraction.pattern_extractor import run_pattern_extractor
from src.services.extraction.pattern_registry import get_registry
from src.services.extraction.persistence import Discrepancy
from src.services.extraction_v3.binding.invariants_runner import run_invariants

log = logging.getLogger(__name__)


def _pipeline_version() -> str:
    """Short git SHA + 'renov' suffix, e.g. 'e8c8563-renov'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "unknown"
    return f"{sha}-renov"


def _serialize_parsed(parsed) -> dict[str, Any]:
    """Build the parser_snapshot JSONB payload. Keep it tight — full_text,
    backend, confidence. Pages/tokens are heavy and not needed for re-grounding
    at the field level (full_text is enough)."""
    return {
        "source_path": getattr(parsed, "source_path", None),
        "file_format": getattr(parsed, "file_format", None),
        "parser_backend": getattr(parsed, "parser_backend", None),
        "parser_confidence": getattr(parsed, "parser_confidence", None),
        "full_text": parsed.full_text,
        "page_count": len(getattr(parsed, "pages", []) or []),
    }


def dispatch_document(
    *, process_monitor_id: int | None, file_path: str, doc_type: str,
) -> dict[str, Any]:
    """Run a single document through the renovation pipeline end-to-end.

    Returns a small result summary: {status, raw_id, doc_pk, n_fields,
    n_discrepancies, trace_id}.
    """
    trace_id = uuid4()
    pipeline_version = _pipeline_version()
    log.info("dispatch start trace=%s doc_type=%s file=%s",
             trace_id, doc_type, file_path)

    # L0
    parsed = parse_document(file_path)

    # L1
    candidates = run_pattern_extractor(parsed, doc_type)

    # L2 — engineered fallbacks for NER-typed fields the L1 regex missed.
    # Only fires for fields with judge.ner_type_check != 'none'.
    registry = get_registry(doc_type)
    l1_fields = {c.field for c in candidates}
    try:
        ner_candidates = fill_ner_gaps(
            parsed=parsed,
            schema=registry.schema,
            existing_fields=l1_fields,
        )
        candidates.extend(ner_candidates)
    except Exception as exc:
        log.warning("NER gap-fill failed: %s", exc)

    # L2 — line item extraction from ParsedDocument.tables
    try:
        line_candidates = extract_line_items(parsed, registry.schema)
        candidates.extend(line_candidates)
    except Exception as exc:
        log.warning("table_extractor failed: %s", exc)

    # L3 — grounded last-resort judge fills required fields that L1+L2 missed.
    # Each judge candidate's evidence_text is verified to be a verbatim
    # substring of parsed.full_text by call_grounded_last_resort itself; the
    # substring gate below re-validates as defence in depth.
    try:
        judge_candidates = run_grounded_judge_for_gaps(
            parsed=parsed,
            registry=registry,
            existing_candidates=candidates,
            file_path=getattr(parsed, "source_path", None),
        )
        candidates.extend(judge_candidates)
    except Exception as exc:
        log.warning("grounded judge gate failed: %s", exc)

    # L3 grounding gate — defence in depth. L1 candidates come from regex
    # over full_text (always substrings); L2 NER spans come from token
    # iteration over the parse (also substrings); L3 judge candidates have
    # already been through call_grounded_last_resort's safety check which
    # uses the same progressive normalisation as below. We replicate the
    # check here so any future candidate source has to honour it too.
    from src.services.extraction_v3.judge.grounded_last_resort import (
        _collapse_letter_spacing, _norm_ws, _WS_RE,
    )
    full_text = parsed.full_text or ""
    full_text_ws = _norm_ws(full_text)
    full_text_sq = _WS_RE.sub("", _collapse_letter_spacing(full_text))

    def _grounded(c) -> bool:
        t = c.span.text
        if not t:
            return False
        if t in full_text:
            return True
        if _norm_ws(t) in full_text_ws:
            return True
        # Letter-spacing collapse + all-whitespace strip — last resort, only
        # accepts evidence whose non-whitespace tokens all appear in the doc.
        if _WS_RE.sub("", _collapse_letter_spacing(t)) in full_text_sq:
            return True
        return False

    grounded = [c for c in candidates if _grounded(c)]

    # Build header record + provenance picks + line items
    columns, picked, bind_errors = persistence.build_header_record(grounded, registry)
    line_items = persistence.build_line_items(grounded, registry)

    # Context understanding layer — single Ollama (BeyondProcwise/AgentNick)
    # pass over the full document text. The fine-tuned procurement model
    # uses the L1/L2/L3 candidates as HINTS but treats full_text as ground
    # truth, disambiguating supplier vs buyer, dropping layout noise, and
    # producing typed/grounded values. Its output REPLACES the regex/judge
    # picks for every schema field — None means "could not be grounded in
    # the document" (no fabrication), which translates to a missing-required
    # discrepancy below if the field is required.
    full_text = parsed.full_text or ""
    if full_text.strip():
        try:
            from src.services.extraction.context_layer import synthesize as _ctx_synth
            synthesized = _ctx_synth(
                doc_type, full_text, columns, file_path=file_path,
            )
            # Build the schema's set of valid db_columns. context_layer's
            # output may include derived fields (exchange_rate_to_usd,
            # converted_amount_usd) that exist on the invoice/PO _raw
            # tables but not the quote _raw table; filtering on the schema
            # keeps the INSERT statement honest.
            # Also include `field.name` for fields with resolves_to_db_column
            # set (e.g. invoice.supplier_name → resolved to supplier_id by
            # supplier_resolver during promote()): the field has db_column=null
            # but the _raw table DOES have a column by that name.
            valid_cols: set[str] = set()
            for f in registry.schema.fields:
                if f.db_column:
                    valid_cols.add(f.db_column)
                if getattr(f, "resolves_to_db_column", None):
                    valid_cols.add(f.name)
            for k, v in synthesized.items():
                if k not in valid_cols:
                    continue
                if v is not None:
                    columns[k] = v
                else:
                    # context_layer says "couldn't ground this" — drop any
                    # regex noise we had. Required fields will surface as
                    # missing_required below; non-required stay NULL.
                    columns.pop(k, None)
        except Exception as exc:  # noqa: BLE001
            log.warning("context_layer synthesis failed: %s", exc)

    discrepancies: list[Discrepancy] = []
    # Re-bind any synthesized values that conflict with column types is
    # already handled by context_layer's _validate_and_bind. Bind errors
    # from the L1 layer are stale once synthesis ran — drop them.
    if not full_text.strip():
        discrepancies.extend(bind_errors)

    # Missing required fields (post-synthesis). columns now reflects the
    # context layer's truth; missing column == AgentNick could not find it.
    for f in registry.schema.fields:
        if f.required and f.db_column and (
            f.db_column not in columns or columns.get(f.db_column) in (None, "")
        ):
            discrepancies.append(Discrepancy(
                field_name=f.name,
                issue_type="missing_required",
                severity="critical",
                blocks_promotion=True,
                notes="context_layer (AgentNick) could not ground a value in the document",
            ))

    # Invariants
    try:
        invariant_results = run_invariants(
            header=columns, line_items=[], schema=registry.schema,
        )
    except Exception as exc:
        log.warning("invariants_runner failed: %s", exc)
        invariant_results = []

    for ir in invariant_results:
        if ir.severity == "CRITICAL":
            discrepancies.append(Discrepancy(
                field_name=ir.name,  # the invariant name; fields_involved not exposed
                issue_type="invariant_failed",
                severity="critical",
                blocks_promotion=True,
                notes=ir.message or f"{ir.name} failed",
            ))
        elif ir.severity == "WARNING":
            discrepancies.append(Discrepancy(
                field_name=ir.name,
                issue_type="invariant_failed",
                severity="warning",
                blocks_promotion=False,
                notes=ir.message or f"{ir.name} warning",
            ))

    blocking = any(d.blocks_promotion for d in discrepancies)
    promotion_status = "discrepancy" if blocking else "pending"

    # `persistence.write_raw` derives doc_pk_candidate from columns[pk_field]
    # internally — since context_layer now writes the authoritative
    # identifier into that schema column (e.g. columns["invoice_id"]),
    # nothing extra is needed here.
    raw_id = persistence.write_raw(
        doc_type=doc_type,
        file_path=file_path,
        process_monitor_id=process_monitor_id,
        trace_id=trace_id,
        pipeline_version=pipeline_version,
        columns=columns,
        parser_snapshot=_serialize_parsed(parsed),
        promotion_status=promotion_status,
    )

    # Write line items (if any)
    if line_items:
        try:
            persistence.write_line_items_raw(
                doc_type=doc_type, raw_id=raw_id, line_items=line_items,
            )
        except Exception as exc:
            log.warning("line_items_raw write failed: %s", exc)

    if discrepancies:
        persistence.write_discrepancies(
            doc_type=doc_type, raw_id=raw_id,
            source_file=file_path,
            doc_pk_candidate=columns.get(persistence._DOC_PK_FIELD[doc_type]),
            discrepancies=discrepancies,
        )

    # Provenance (only when we have a doc_pk — required by the table's NOT NULL)
    persistence.write_provenance(
        doc_type=doc_type,
        doc_pk=columns.get(persistence._DOC_PK_FIELD[doc_type]),
        pipeline_version=pipeline_version,
        picked=picked,
        registry=registry,
    )

    doc_pk = columns.get(persistence._DOC_PK_FIELD[doc_type])

    # Auto-promote when clean (no blocking discrepancy) and we have a doc_pk.
    # Discrepancy rows wait for HITL → trigger → listener.
    final_status = promotion_status
    if not blocking and doc_pk:
        prom = promotion.promote(raw_id, doc_type)
        if prom.get("ok"):
            final_status = "promoted"
        else:
            log.warning("inline promote failed: %s", prom.get("reason"))
            final_status = "pending"  # _raw kept; manual retry possible
    elif not blocking and not doc_pk:
        # Missing PK is a soft failure — keep the _raw row for inspection,
        # mark status='no_pk' so it doesn't sit in 'pending' indefinitely.
        persistence.update_promotion_status(
            doc_type=doc_type, raw_id=raw_id, status="no_pk",
        )
        final_status = "no_pk"

    missing_required_fields = sorted({
        d.field_name for d in discrepancies
        if d.issue_type == "missing_required" and d.blocks_promotion
    })
    result = {
        "status": final_status,
        "raw_id": raw_id,
        "doc_pk": doc_pk,
        "n_fields": len(columns),
        # Watcher reads this key (legacy v3 shape) to decide whether to log
        # the ZERO_LINE_ITEMS warning. Surface the count we actually wrote
        # so the warning fires only when extraction genuinely missed lines,
        # not when this newer pipeline simply doesn't include the key.
        "line_items": len(line_items),
        "header_fields": len(columns),
        "n_discrepancies": len(discrepancies),
        "discrepancies": len(discrepancies),
        "missing_required": missing_required_fields,
        "trace_id": str(trace_id),
        "pipeline_version": pipeline_version,
        "raw_persisted": True,
    }
    log.info("dispatch end %s", result)
    return result
