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
introduced. The endpoints are unauthenticated for parity with the rest
of the API; an upstream reverse-proxy is expected to gate access.
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
    invoices, quotes, and POs in proc.bp_*."""
    sql = """
    WITH
    invoices AS (
        SELECT supplier_id,
               invoice_id::text AS pk,
               COALESCE(invoice_total_incl_tax, 0)::float AS total,
               (SELECT COUNT(*) FROM proc.bp_invoice_line_items li
                  WHERE li.invoice_id = bp.invoice_id) AS lines,
               'Invoice' AS doc_type,
               GREATEST(created_date, last_modified_date) AS seen_at
          FROM proc.bp_invoice bp
    ),
    quotes AS (
        SELECT supplier_id,
               quote_id::text AS pk,
               COALESCE(total_amount, 0)::float AS total,
               (SELECT COUNT(*) FROM proc.bp_quote_line_items li
                  WHERE li.quote_id = bp.quote_id) AS lines,
               'Quote' AS doc_type,
               GREATEST(created_date, last_modified_date) AS seen_at
          FROM proc.bp_quote bp
    ),
    pos AS (
        SELECT supplier_id,
               po_id::text AS pk,
               COALESCE(total_amount, 0)::float AS total,
               (SELECT COUNT(*) FROM proc.bp_po_line_items li
                  WHERE li.po_id = bp.po_id) AS lines,
               'Purchase_Order' AS doc_type,
               GREATEST(created_date, last_modified_date) AS seen_at
          FROM proc.bp_purchase_order bp
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
    WITH unioned AS (
        SELECT 'Invoice' AS doc_type, invoice_id::text AS pk, supplier_id,
               COALESCE(invoice_total_incl_tax, 0)::float AS total,
               (SELECT COUNT(*) FROM proc.bp_invoice_line_items li
                  WHERE li.invoice_id = bp.invoice_id) AS lines,
               GREATEST(created_date, last_modified_date) AS seen_at
          FROM proc.bp_invoice bp
        UNION ALL
        SELECT 'Quote', quote_id::text, supplier_id,
               COALESCE(total_amount, 0)::float,
               (SELECT COUNT(*) FROM proc.bp_quote_line_items li
                  WHERE li.quote_id = bp.quote_id),
               GREATEST(created_date, last_modified_date)
          FROM proc.bp_quote bp
        UNION ALL
        SELECT 'Purchase_Order', po_id::text, supplier_id,
               COALESCE(total_amount, 0)::float,
               (SELECT COUNT(*) FROM proc.bp_po_line_items li
                  WHERE li.po_id = bp.po_id),
               GREATEST(created_date, last_modified_date)
          FROM proc.bp_purchase_order bp
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
