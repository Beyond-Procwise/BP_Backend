"""Extraction telemetry service — captures per-document extraction quality.

For every document that the extraction pipeline has processed (a terminal row in
proc.process_monitor: Extracted / Extraction_Failed), this derives a structured
quality record from the already-persisted truth — the _stg row (extracted +
computed columns), proc.bp_extraction_discrepancy (gaps/source errors), and the
process_monitor row (upload + vendor pattern) — and writes it to
proc.bp_extraction_telemetry.

It does NOT touch the extraction pipeline (no procwise restart needed). It runs
standalone, either as a one-shot backfill or as a polling daemon that captures
each newly-processed upload. The telemetry table is the analysis surface for
"how well is extraction doing, what patterns, what gaps" → drives improvement.

Usage:
    .venv/bin/python -m src.services.extraction_telemetry.telemetry_service --once
    .venv/bin/python -m src.services.extraction_telemetry.telemetry_service        # daemon
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

from src.services.db import get_conn
from src.services.extraction.completeness import assess, _SUBTOTAL_COL  # noqa

log = logging.getLogger("extraction_telemetry")

POLL_SECONDS = float(os.getenv("TELEMETRY_POLL_SECONDS", "60"))

# category (process_monitor.category) -> (doc_type, stg table, pk col, line table, line fk)
_CATEGORY = {
    "invoice": ("invoice", "bp_invoice_stg", "invoice_id",
                "bp_invoice_line_items_stg", "invoice_id"),
    "quote": ("quote", "bp_quote_stg", "quote_id",
              "bp_quote_line_items_stg", "quote_id"),
    "quotes": ("quote", "bp_quote_stg", "quote_id",
               "bp_quote_line_items_stg", "quote_id"),
    "po": ("purchase_order", "bp_purchase_order_stg", "po_id",
           "bp_po_line_items_stg", "po_id"),
    "purchase_order": ("purchase_order", "bp_purchase_order_stg", "po_id",
                       "bp_po_line_items_stg", "po_id"),
}

_WS = re.compile(r"\s+")
_VENDOR = re.compile(r"^([A-Za-z][A-Za-z&,\.\s]{1,40}?)\s+(?:INV|PO|QUT|QTE|QUOTE)", re.I)


def _sq(s) -> str:
    return _WS.sub("", str(s or "")).lower()


def _vendor_hint(file_path: str) -> str | None:
    """Leading supplier token of the filename — a coarse layout/vendor pattern key.

    Delegates to the shared vendor_key so the feedback proposer (producer) and
    context_layer (consumer) derive the identical key. Behaviour-preserving.
    """
    from src.services.extraction_feedback.vendor_key import vendor_key
    return vendor_key(file_path)


def _stg_columns(cur, table: str) -> list[str]:
    cur.execute(
        "select column_name from information_schema.columns "
        "where table_schema='proc' and table_name=%s", (table,))
    return [r[0] for r in cur.fetchall()]


def _match_pk(cur, stg_table: str, pk_col: str, file_path: str) -> str | None:
    """Find the _stg PK whose normalized form appears in the filename.
    Robust to INV/PO/QUT prefix + separator differences (squeezed match)."""
    fn = _sq(os.path.basename(file_path))
    cur.execute(f"select {pk_col} from proc.{stg_table}")
    for (pk,) in cur.fetchall():
        if pk and _sq(pk) in fn:
            return pk
    return None


def _discrepancy_summary(cur, doc_pk: str | None, source_file: str) -> tuple[int, dict]:
    """(total, {issue_type: count}) for this document from bp_extraction_discrepancy."""
    cur.execute(
        "select issue_type, count(*) from proc.bp_extraction_discrepancy "
        "where (doc_pk_candidate = %s and %s <> '') or source_file = %s "
        "group by issue_type",
        (doc_pk, doc_pk or "", source_file),
    )
    by_type: dict[str, int] = {}
    for issue_type, n in cur.fetchall():
        by_type[issue_type or "unknown"] = by_type.get(issue_type or "unknown", 0) + int(n)
    return sum(by_type.values()), by_type


def compute_record(cur, pm: dict) -> dict:
    """Build one telemetry record for a terminal process_monitor row."""
    file_path = pm.get("file_path") or ""
    category = (pm.get("category") or "").lower()
    spec = _CATEGORY.get(category)
    rec = {
        "process_monitor_id": pm.get("id"),
        "doc_type": spec[0] if spec else category,
        "file_path": file_path,
        "vendor_hint": _vendor_hint(file_path),
        "doc_pk": None,
        "status": pm.get("status"),
        "completeness_status": None,
        "confidence": None,
        "header_fields": None,
        "line_items": None,
        "n_discrepancies": 0,
        "discrepancy_types": "{}",
        "missing_required": None,
        "currency": None,
        "converted_amount_usd": None,
        "parser_backend": None,   # only known at extraction time; null post-hoc
        "page_count": None,
        "pipeline_version": None,
        "trace_id": None,
        "error_detail": None,
        "notes": None,
    }
    if pm.get("status") == "Extraction_Failed":
        rec["notes"] = "extraction failed — no _stg row expected"
        n, types = _discrepancy_summary(cur, None, file_path)
        rec["n_discrepancies"] = n
        rec["discrepancy_types"] = json.dumps(types)
        return rec
    if not spec:
        rec["notes"] = f"unmapped category={category!r}"
        return rec

    doc_type, stg_table, pk_col, line_table, line_fk = spec
    pk = _match_pk(cur, stg_table, pk_col, file_path)
    rec["doc_pk"] = pk
    if pk is None:
        rec["completeness_status"] = "no_stg_row"
        rec["notes"] = "marked Extracted but no matching _stg row found (gap)"
        n, types = _discrepancy_summary(cur, None, file_path)
        rec["n_discrepancies"] = n
        rec["discrepancy_types"] = json.dumps(types)
        return rec

    cols = _stg_columns(cur, stg_table)
    cur.execute(f"select * from proc.{stg_table} where {pk_col} = %s", (pk,))
    row = dict(zip(cols, cur.fetchone()))
    # line items — fetch column names BEFORE the data query, otherwise the
    # information_schema lookup overwrites/consumes the cursor result set.
    lcols = _stg_columns(cur, line_table)
    cur.execute(f"select * from proc.{line_table} where {line_fk} = %s", (pk,))
    lines = [dict(zip(lcols, r)) for r in cur.fetchall()]

    # required fields missing? use the schema via completeness has_line_schema=True
    rec["confidence"] = row.get("confidence_score")
    rec["currency"] = row.get("currency")
    rec["converted_amount_usd"] = row.get("converted_amount_usd")
    rec["line_items"] = len(lines)
    audit_cols = {"created_date", "created_by", "last_modified_by", "last_modified_date",
                  "confidence_score", "trigger_type", "trigger_context_description",
                  "ai_flag_required"}
    rec["header_fields"] = sum(
        1 for k, v in row.items() if k not in audit_cols and v not in (None, ""))
    report = assess(doc_type, row, lines, has_line_schema=True,
                    missing_required=[])
    rec["completeness_status"] = report.status
    n, types = _discrepancy_summary(cur, pk, file_path)
    rec["n_discrepancies"] = n
    rec["discrepancy_types"] = json.dumps(types)
    return rec


_INSERT = """
INSERT INTO proc.bp_extraction_telemetry
 (process_monitor_id, doc_type, file_path, vendor_hint, doc_pk, status,
  completeness_status, confidence, header_fields, line_items, n_discrepancies,
  discrepancy_types, missing_required, currency, converted_amount_usd,
  parser_backend, page_count, pipeline_version, trace_id, error_detail, notes)
VALUES
 (%(process_monitor_id)s, %(doc_type)s, %(file_path)s, %(vendor_hint)s, %(doc_pk)s,
  %(status)s, %(completeness_status)s, %(confidence)s, %(header_fields)s,
  %(line_items)s, %(n_discrepancies)s, %(discrepancy_types)s::jsonb,
  %(missing_required)s, %(currency)s, %(converted_amount_usd)s, %(parser_backend)s,
  %(page_count)s, %(pipeline_version)s, %(trace_id)s, %(error_detail)s, %(notes)s)
"""


def capture_new(conn) -> int:
    """Capture telemetry for terminal process_monitor docs (re)processed since
    their last telemetry row. Returns number of new telemetry rows written."""
    written = 0
    with conn.cursor() as cur:
        # Docs in a terminal state whose most-recent processing is newer than the
        # latest telemetry capture for that process_monitor_id (or never captured).
        cur.execute("""
            select pm.id, pm.process_name, pm.status, pm.file_path, pm.category,
                   pm.lastmodified_date, pm.end_ts
              from proc.process_monitor pm
             where pm.status in ('Extracted','Extraction_Failed')
               and not exists (
                   select 1 from proc.bp_extraction_telemetry t
                    where t.process_monitor_id = pm.id
                      and t.captured_at >= coalesce(pm.end_ts, pm.lastmodified_date, now())
               )
             order by pm.id
        """)
        cols = [d.name for d in cur.description]
        pend = [dict(zip(cols, r)) for r in cur.fetchall()]

    for pm in pend:
        try:
            with conn.cursor() as cur:
                rec = compute_record(cur, pm)
                cur.execute(_INSERT, rec)
                written += 1
        except Exception:
            log.exception("telemetry capture failed for process_monitor_id=%s", pm.get("id"))
    return written


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="backfill once and exit")
    args = ap.parse_args()

    if args.once:
        with get_conn() as conn:
            n = capture_new(conn)
        log.info("telemetry backfill: wrote %d record(s)", n)
        return 0

    log.info("extraction telemetry service started (poll=%.0fs)", POLL_SECONDS)
    while True:
        try:
            with get_conn() as conn:
                n = capture_new(conn)
            if n:
                log.info("captured %d new extraction telemetry record(s)", n)
        except Exception:
            log.exception("telemetry poll failed")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
