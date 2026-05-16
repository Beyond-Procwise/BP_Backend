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

from src.services.extraction import persistence
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

    # L3 — substring grounding gate (all candidates must already be substrings
    # by construction, but defensive enforce here so any future source that
    # bypasses construction still gets caught)
    grounded = [c for c in candidates if c.span.text in parsed.full_text]

    registry = get_registry(doc_type)

    # Build header record + provenance picks
    columns, picked, bind_errors = persistence.build_header_record(grounded, registry)

    discrepancies: list[Discrepancy] = list(bind_errors)

    # Missing required fields
    for f in registry.schema.fields:
        if f.required and f.db_column and f.db_column not in columns:
            discrepancies.append(Discrepancy(
                field_name=f.name,
                issue_type="missing_required",
                severity="critical",
                blocks_promotion=True,
                notes=f"no candidate above threshold {f.confidence_threshold}",
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

    result = {
        "status": promotion_status,
        "raw_id": raw_id,
        "doc_pk": columns.get(persistence._DOC_PK_FIELD[doc_type]),
        "n_fields": len(columns),
        "n_discrepancies": len(discrepancies),
        "trace_id": str(trace_id),
        "pipeline_version": pipeline_version,
    }
    log.info("dispatch end %s", result)
    return result
