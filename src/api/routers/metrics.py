"""Read-only extraction-quality metrics.

Surface for operators to see how each vendor is performing without
having to query the database directly. Three endpoints:

  GET /metrics/extraction
      Per-vendor: total processed, success_rate, mean_confidence,
      zero_lines_rate, template_applied_rate, last_seen_at.

  GET /metrics/extraction/recent?limit=N
      Last N processed records with their pk, supplier, line-count,
      confidence, status — newest first. Useful for live dashboards.

  GET /metrics/extraction/templates
      Inventory of vendor templates currently in proc.bp_extraction_template
      with their hint counts and success counts.

All queries are READ-ONLY against the existing tables; no new schema is
introduced. The router is mounted in `_AUTHENTICATED_ROUTERS` (api/main.py),
so every endpoint here requires a verified principal under ASK_AUTH_MODE
-- the docstring previously claimed the opposite and had been wrong since
that list was introduced.

WHICH TABLES, AND WHY IT MATTERS

The document queries read the `_trgt` tables. The pipeline writes
`bp_<doc>_raw` -> `bp_<doc>_stg` -> `bp_<doc>_trgt`, and `_trgt` is the
final, deal-keyed destination -- the only stage that represents what the
product actually believes about a document.

These queries originally named `proc.bp_invoice`, `proc.bp_quote` and
`proc.bp_purchase_order`, which have never existed in this schema, so two
of the three endpoints returned 500 for months. Nothing caught it: there
was no test for this router, and the output-safety layer turned the
`UndefinedTable` error into a generic message, so it read as a transient
server fault rather than a query naming a table that is not there.

WHY THE LINE COUNTS ARE PRE-AGGREGATED

`lines` used to be a correlated subquery evaluated once per document. None
of the `*_line_items_trgt` tables carries an index, so each of ~38.5k parent
rows sequentially scanned a 55k-116k row table: EXPLAIN cost 161,402,845,
and the query had not returned after 175 seconds. Counting once per table in
a CTE and LEFT JOINing costs 10,324 and returns in 0.2s. Keep it that way --
and note that adding the missing indexes would make the old shape *look*
acceptable while remaining quadratic in the corpus.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


def _query(sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read-only query via the shared connection factory."""
    from services.db import get_conn
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


@router.get(
    "/extraction",
    summary="Per-vendor extraction quality summary",
)
def extraction_quality_summary() -> dict[str, Any]:
    """Aggregate quality metrics, grouped by supplier_id, across all
    invoices, quotes, and POs in the proc.bp_*_trgt tables."""
    sql = """
    WITH
    invoice_lines AS (
        SELECT invoice_id, COUNT(*) AS lines
          FROM proc.bp_invoice_line_items_trgt GROUP BY invoice_id
    ),
    quote_lines AS (
        SELECT quote_id, COUNT(*) AS lines
          FROM proc.bp_quote_line_items_trgt GROUP BY quote_id
    ),
    po_lines AS (
        SELECT po_id, COUNT(*) AS lines
          FROM proc.bp_po_line_items_trgt GROUP BY po_id
    ),
    invoices AS (
        SELECT bp.supplier_id,
               COALESCE(bp.invoice_total_incl_tax, 0)::float AS total,
               COALESCE(l.lines, 0) AS lines,
               'Invoice' AS doc_type,
               GREATEST(bp.created_date, bp.last_modified_date) AS seen_at
          FROM proc.bp_invoice_trgt bp
          LEFT JOIN invoice_lines l ON l.invoice_id = bp.invoice_id
    ),
    quotes AS (
        SELECT bp.supplier_id,
               COALESCE(bp.total_amount, 0)::float AS total,
               COALESCE(l.lines, 0) AS lines,
               'Quote' AS doc_type,
               GREATEST(bp.created_date, bp.last_modified_date) AS seen_at
          FROM proc.bp_quote_trgt bp
          LEFT JOIN quote_lines l ON l.quote_id = bp.quote_id
    ),
    pos AS (
        SELECT bp.supplier_id,
               COALESCE(bp.total_amount, 0)::float AS total,
               COALESCE(l.lines, 0) AS lines,
               'Purchase_Order' AS doc_type,
               GREATEST(bp.created_date, bp.last_modified_date) AS seen_at
          FROM proc.bp_purchase_order_trgt bp
          LEFT JOIN po_lines l ON l.po_id = bp.po_id
    ),
    union_all AS (
        SELECT * FROM invoices
        UNION ALL SELECT * FROM quotes
        UNION ALL SELECT * FROM pos
    )
    SELECT
        supplier_id,
        doc_type,
        COUNT(*) AS processed,
        SUM(CASE WHEN total > 0 AND lines = 0 THEN 1 ELSE 0 END) AS zero_lines_with_total,
        SUM(CASE WHEN lines >= 1 THEN 1 ELSE 0 END) AS with_lines,
        MAX(seen_at) AS last_seen_at
      FROM union_all
     GROUP BY supplier_id, doc_type
     ORDER BY processed DESC, supplier_id
    """
    try:
        rows = _query(sql)
    except Exception as exc:
        logger.exception("metrics query failed")
        raise HTTPException(status_code=500, detail=f"metrics query failed: {exc}")

    vendors: list[dict[str, Any]] = []
    totals = {"processed": 0, "with_lines": 0, "zero_lines_with_total": 0}
    for supplier_id, doc_type, processed, zero_with_total, with_lines, last_seen in rows:
        zero_rate = (zero_with_total / processed) if processed else 0.0
        line_rate = (with_lines / processed) if processed else 0.0
        vendors.append({
            "supplier_id": supplier_id,
            "doc_type": doc_type,
            "processed": int(processed),
            "with_lines": int(with_lines),
            "zero_lines_with_total": int(zero_with_total),
            "zero_lines_rate": round(zero_rate, 3),
            "line_capture_rate": round(line_rate, 3),
            "last_seen_at": str(last_seen) if last_seen else None,
        })
        totals["processed"] += int(processed)
        totals["with_lines"] += int(with_lines)
        totals["zero_lines_with_total"] += int(zero_with_total)

    grand = {
        "processed": totals["processed"],
        "with_lines": totals["with_lines"],
        "zero_lines_with_total": totals["zero_lines_with_total"],
        "line_capture_rate": round(
            totals["with_lines"] / totals["processed"], 3
        ) if totals["processed"] else 0.0,
        "zero_lines_rate": round(
            totals["zero_lines_with_total"] / totals["processed"], 3
        ) if totals["processed"] else 0.0,
    }
    return {"totals": grand, "vendors": vendors}


@router.get(
    "/extraction/recent",
    summary="Recently-processed records (newest first)",
)
def recent_extractions(
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    sql = """
    WITH
    invoice_lines AS (
        SELECT invoice_id, COUNT(*) AS lines
          FROM proc.bp_invoice_line_items_trgt GROUP BY invoice_id
    ),
    quote_lines AS (
        SELECT quote_id, COUNT(*) AS lines
          FROM proc.bp_quote_line_items_trgt GROUP BY quote_id
    ),
    po_lines AS (
        SELECT po_id, COUNT(*) AS lines
          FROM proc.bp_po_line_items_trgt GROUP BY po_id
    ),
    unioned AS (
        SELECT 'Invoice' AS doc_type, bp.invoice_id::text AS pk, bp.supplier_id,
               COALESCE(bp.invoice_total_incl_tax, 0)::float AS total,
               COALESCE(l.lines, 0) AS lines,
               GREATEST(bp.created_date, bp.last_modified_date) AS seen_at
          FROM proc.bp_invoice_trgt bp
          LEFT JOIN invoice_lines l ON l.invoice_id = bp.invoice_id
        UNION ALL
        SELECT 'Quote', bp.quote_id::text, bp.supplier_id,
               COALESCE(bp.total_amount, 0)::float,
               COALESCE(l.lines, 0),
               GREATEST(bp.created_date, bp.last_modified_date)
          FROM proc.bp_quote_trgt bp
          LEFT JOIN quote_lines l ON l.quote_id = bp.quote_id
        UNION ALL
        SELECT 'Purchase_Order', bp.po_id::text, bp.supplier_id,
               COALESCE(bp.total_amount, 0)::float,
               COALESCE(l.lines, 0),
               GREATEST(bp.created_date, bp.last_modified_date)
          FROM proc.bp_purchase_order_trgt bp
          LEFT JOIN po_lines l ON l.po_id = bp.po_id
    )
    SELECT doc_type, pk, supplier_id, total, lines, seen_at
      FROM unioned
     ORDER BY seen_at DESC NULLS LAST
     LIMIT %s
    """
    try:
        rows = _query(sql, (limit,))
    except Exception as exc:
        logger.exception("metrics recent query failed")
        raise HTTPException(status_code=500, detail=f"metrics query failed: {exc}")
    return {
        "records": [
            {
                "doc_type": r[0],
                "pk": r[1],
                "supplier_id": r[2],
                "total": float(r[3] or 0),
                "lines": int(r[4] or 0),
                "needs_review": bool(float(r[3] or 0) > 0 and int(r[4] or 0) == 0),
                "seen_at": str(r[5]) if r[5] else None,
            } for r in rows
        ],
    }


@router.get(
    "/extraction/templates",
    summary="Vendor templates currently active",
)
def template_inventory() -> dict[str, Any]:
    sql = """
    SELECT vendor_name, doc_type,
           SUBSTRING(fingerprint, 1, 12) AS fp_prefix,
           jsonb_array_length(jsonb_path_query_array(field_hints, '$.*')) AS field_hint_count,
           CASE WHEN line_item_hints IS NULL THEN 0
                ELSE jsonb_array_length(jsonb_path_query_array(line_item_hints->'column_map', '$.*')) END AS line_col_count,
           success_count, correction_count,
           created_at, last_used_at
      FROM proc.bp_extraction_template
     ORDER BY vendor_name
    """
    try:
        rows = _query(sql)
    except Exception as exc:
        logger.exception("templates query failed")
        raise HTTPException(status_code=500, detail=f"metrics query failed: {exc}")
    return {
        "templates": [
            {
                "vendor_name": r[0],
                "doc_type": r[1],
                "fingerprint_prefix": r[2],
                "field_hint_count": int(r[3] or 0),
                "line_item_columns": int(r[4] or 0),
                "success_count": int(r[5] or 0),
                "correction_count": int(r[6] or 0),
                "created_at": str(r[7]) if r[7] else None,
                "last_used_at": str(r[8]) if r[8] else None,
            } for r in rows
        ],
    }
