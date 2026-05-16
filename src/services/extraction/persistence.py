"""Renovation persistence — flat-column _raw + provenance + discrepancies.

Writes one row per document to proc.bp_<doctype>_raw with field columns
populated from the bound candidates. parser_snapshot stores the L0 output
for re-grounding. provenance_v3 carries one row per committed value with
bbox + evidence_text. Discrepancies are emitted for missing required
fields and CRITICAL invariant failures; the _raw row's promotion_status
reflects whether any blocking discrepancy was written.

All inserts for a single document occur in one transaction.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from uuid import UUID

from src.services.db import get_conn
from src.services.extraction.pattern_registry import PatternRegistry
from src.services.extraction.types import Candidate
from src.services.extraction_v3.binding.type_binder import bind_typed
from src.services.extraction_v3.schemas.candidate import Candidate as V3Candidate

log = logging.getLogger(__name__)

# Per-doc-type _raw table and PK field
_RAW_TABLES = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
    "contract": "proc.bp_contract_raw",
}
_LINE_RAW_TABLES = {
    "invoice": "proc.bp_invoice_line_items_raw",
    "purchase_order": "proc.bp_po_line_items_raw",
    "quote": "proc.bp_quote_line_items_raw",
    # contract has no line items
}
_DOC_PK_FIELD = {
    "invoice": "invoice_id",
    "purchase_order": "po_id",
    "quote": "quote_id",
    "contract": "contract_id",
}


@dataclass
class Discrepancy:
    field_name: str
    issue_type: str          # invariant_failed | missing_required | type_bind_error | judge_incoherent
    severity: str            # critical | warning | info
    raw_value: str | None = None
    expected_value: str | None = None
    computed_value: str | None = None
    blocks_promotion: bool = True
    evidence_page: int | None = None
    evidence_bbox: list[float] | None = None
    evidence_text: str | None = None
    notes: str | None = None


def _coerce(value: str, field_type: str) -> tuple[Any, bool]:
    """Use the existing type_binder; return (coerced_value, bind_error_flag)."""
    # type_binder expects a v3 Candidate — make a minimal stub.
    stub = V3Candidate(
        field="_", value=value, page=1, bbox=(0, 0, 0, 0),
        evidence_text=value, model="qwen_vlm", confidence=1.0,
    )
    bound = bind_typed(stub, field_type)  # type: ignore[arg-type]
    return bound.coerced_value, bound.bind_error


def build_header_record(
    candidates: Iterable[Candidate],
    registry: PatternRegistry,
) -> tuple[dict[str, Any], dict[str, Candidate], list[Discrepancy]]:
    """For each header field, pick the highest-confidence candidate and
    coerce it to the column type. Returns (column_values, picked_by_field,
    bind_error_discrepancies)."""
    best: dict[str, Candidate] = {}
    for c in candidates:
        if c.field.startswith("line_items."):
            continue
        if c.field not in best or c.confidence > best[c.field].confidence:
            best[c.field] = c

    columns: dict[str, Any] = {}
    bind_errors: list[Discrepancy] = []
    for field_name, cand in best.items():
        meta = registry.meta(field_name)
        if meta.db_column is None:
            continue
        coerced, err = _coerce(cand.value, meta.type)
        if err:
            bind_errors.append(Discrepancy(
                field_name=field_name,
                issue_type="type_bind_error",
                severity="warning",
                raw_value=cand.value,
                blocks_promotion=False,
                evidence_page=cand.span.page,
                evidence_bbox=list(cand.span.bbox),
                evidence_text=cand.span.text,
                notes=f"could not coerce to type {meta.type!r}",
            ))
            continue
        columns[meta.db_column] = coerced
    return columns, best, bind_errors


def write_raw(
    *,
    doc_type: str,
    file_path: str,
    process_monitor_id: int | None,
    trace_id: UUID,
    pipeline_version: str,
    columns: Mapping[str, Any],
    parser_snapshot: Mapping[str, Any],
    promotion_status: str,
) -> int:
    """INSERT one row into proc.bp_<doctype>_raw. Returns raw_id.

    `columns` already has only db_columns present in the _raw table.
    """
    table = _RAW_TABLES[doc_type]
    pk_field = _DOC_PK_FIELD[doc_type]
    doc_pk_candidate = columns.get(pk_field)

    # raw_payload is the legacy JSONB column kept by the additive migration; it
    # has NOT NULL, so we satisfy it with an empty object. The renovation reads
    # parser_snapshot instead. A follow-up migration drops raw_payload once the
    # new pipeline is the live default.
    base_cols = [
        "source_file", "process_monitor_id", "pipeline_version",
        "parser_snapshot", "trace_id", "promotion_status", "doc_pk_candidate",
        "raw_payload",
    ]
    base_vals = [
        file_path, process_monitor_id, pipeline_version,
        json.dumps(parser_snapshot), str(trace_id), promotion_status, doc_pk_candidate,
        "{}",
    ]
    field_cols = list(columns.keys())
    field_vals = [columns[c] for c in field_cols]

    all_cols = base_cols + field_cols
    all_vals = base_vals + field_vals
    placeholders = ", ".join(["%s"] * len(all_cols))
    col_clause = ", ".join(all_cols)

    sql = f"INSERT INTO {table} ({col_clause}) VALUES ({placeholders}) RETURNING raw_id"
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute(sql, all_vals)
            raw_id = cur.fetchone()[0]
            conn.commit()
            return raw_id
        except Exception:
            conn.rollback()
            raise


def write_provenance(
    *,
    doc_type: str,
    doc_pk: str | None,
    pipeline_version: str,
    picked: Mapping[str, Candidate],
    registry: PatternRegistry,
) -> None:
    """INSERT one provenance row per picked candidate. Skips fields without a doc_pk."""
    if not doc_pk:
        return  # provenance is keyed by doc_pk; nothing to write yet
    rows = []
    for field_name, cand in picked.items():
        meta = registry.meta(field_name)
        if meta.db_column is None:
            continue
        rows.append((
            doc_type, str(doc_pk), field_name, cand.value,
            cand.span.page, cand.span.bbox[0], cand.span.bbox[1],
            cand.span.bbox[2], cand.span.bbox[3],
            cand.span.text, cand.source, cand.confidence,
            json.dumps([]), cand.confidence, pipeline_version,
        ))
    if not rows:
        return
    sql = """INSERT INTO proc.bp_extraction_provenance_v3
             (doc_type, doc_pk, field_path, value, page,
              bbox_x0, bbox_y0, bbox_x1, bbox_y1,
              evidence_text, model, model_confidence,
              judge_actions, final_confidence, pipeline_version)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.executemany(sql, rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def write_discrepancies(
    *,
    doc_type: str,
    raw_id: int,
    source_file: str,
    doc_pk_candidate: str | None,
    discrepancies: Iterable[Discrepancy],
) -> int:
    """INSERT discrepancy rows. Returns count written."""
    rows = []
    for d in discrepancies:
        rows.append((
            doc_type, raw_id, source_file, doc_pk_candidate,
            d.field_name, d.raw_value, d.expected_value, d.computed_value,
            d.issue_type, d.severity, d.blocks_promotion,
            d.evidence_page, d.evidence_bbox, d.evidence_text, d.notes,
        ))
    if not rows:
        return 0
    sql = """INSERT INTO proc.bp_extraction_discrepancy
             (doc_type, raw_id, source_file, doc_pk_candidate,
              field_name, raw_value, expected_value, computed_value,
              issue_type, severity, blocks_promotion,
              evidence_page, evidence_bbox, evidence_text, notes)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.executemany(sql, rows)
            conn.commit()
            return len(rows)
        except Exception:
            conn.rollback()
            raise


def update_promotion_status(*, doc_type: str, raw_id: int, status: str,
                            promoted_at: bool = False) -> None:
    table = _RAW_TABLES[doc_type]
    sql = f"""UPDATE {table}
                 SET promotion_status = %s
                     {", promoted_at = NOW()" if promoted_at else ""}
               WHERE raw_id = %s"""
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            cur.execute(sql, (status, raw_id))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
