"""Live-data wiring for the deterministic benchmark pricing engine.

Loads a deal's quote lines and the pooled PO/invoice price history from
bp_sqldb (_trgt tables), normalises the match keys, and runs
services.benchmark.engine.compute_benchmark over each line.

The ENGINE stays exact-match and pure; all live-data messiness is handled
HERE, explicitly and disclosed in the response:

- item/uom/currency are normalised (whitespace/case; NULL uom -> "each",
  NULL currency -> "GBP") before they become match keys.
- The corpus has no spec scores, SLA scores, location cost indices or price
  indices, so those adjustments are NEUTRALISED (factor 1.0) by feeding the
  engine identical values on both sides — never fabricated data. Volume is
  the only live adjustment (historical quantities are real).
- No Location Index / Index tables exist in the DB yet, so lookups run
  against empty tables and the engine records "location_default" /
  "index_default" in fallbacks_used (Decision 3 behaviour, visible live).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from services.benchmark.engine import compute_benchmark
from services.benchmark.models import BenchmarkPoint, BenchmarkSettings, QuoteLine

logger = logging.getLogger(__name__)

# Neutral midpoint used for BOTH the quote's requested scores and every
# benchmark point's scores: identical values => spec/SLA factors == 1.0.
_NEUTRAL_SCORE = 5.0

# Written in product terms (the output-safety gate rewrites anything that
# reads like backend internals — e.g. slash-separated words resemble routes).
DISCLOSURES = [
    "specification, service-level, location and inflation adjustments are neutral (no scoring data captured yet); volume reflects real purchase history",
    "match keys normalised: whitespace and case folded; missing unit defaults to 'each', missing currency to 'GBP'",
    "price history excludes this deal's own purchase orders and invoices, so a supplier is never compared against its own price",
    "no location or market index reference data captured yet: neutral defaults applied and recorded on every line",
    "some comparison prices come from documents with an unresolved data-quality finding; each line reports how many",
]


def _norm_item(value: Optional[str]) -> str:
    return " ".join((value or "").split()).lower()


def _norm_uom(value: Optional[str]) -> str:
    return (value or "").strip().lower() or "each"


def _norm_currency(value: Optional[str]) -> str:
    return (value or "").strip().upper() or "GBP"


def load_quote_lines(cur, deal_id: str) -> list[dict[str, Any]]:
    """Quote lines for one deal, with header currency/country/region."""
    cur.execute(
        """
        SELECT q.quote_line_id, q.quote_id, q.item_description, q.quantity,
               q.unit_price, q.unit_of_measure,
               COALESCE(q.currency, h.currency) AS currency,
               h.country, h.region
        FROM proc.bp_quote_line_items_trgt q
        LEFT JOIN proc.bp_quote_trgt h ON h.quote_id = q.quote_id
        WHERE q.deal_id = %s AND q.unit_price IS NOT NULL
        ORDER BY q.quote_line_id
        """,
        (deal_id,),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def load_benchmark_pool(cur, exclude_deal_id: Optional[str] = None) -> list[dict[str, Any]]:
    """All PO + invoice lines with a price — the price-history pool.

    Currency is taken from the document header when the line does not carry it.
    Every live PO line has a NULL currency while 57% of PO headers are not
    sterling, so reading the line alone silently relabels dollars as pounds.
    """
    cur.execute(
        """
        SELECT 'po:' || p.po_line_id AS point_id, p.item_description,
               p.unit_of_measure, COALESCE(p.currency, h.currency) AS currency,
               p.unit_price, p.quantity, p.po_id AS doc_id
        FROM proc.bp_po_line_items_trgt p
        LEFT JOIN proc.bp_purchase_order_trgt h ON h.po_id = p.po_id
        WHERE p.unit_price IS NOT NULL AND p.item_description IS NOT NULL
          AND (%(deal)s::text IS NULL OR p.deal_id IS DISTINCT FROM %(deal)s)
        UNION ALL
        SELECT 'inv:' || i.invoice_line_id, i.item_description,
               i.unit_of_measure, h.currency, i.unit_price, i.quantity,
               i.invoice_id AS doc_id
        FROM proc.bp_invoice_line_items_trgt i
        LEFT JOIN proc.bp_invoice_trgt h ON h.invoice_id = i.invoice_id
        WHERE i.unit_price IS NOT NULL AND i.item_description IS NOT NULL
          AND (%(deal)s::text IS NULL OR i.deal_id IS DISTINCT FROM %(deal)s)
        """,
        {"deal": exclude_deal_id},
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _pool_delta(full: int, scoped: int) -> int:
    """How many points the deal's own documents contributed."""
    return max(0, full - scoped)


def load_flagged_documents(cur) -> set[str]:
    """Document ids carrying at least one open finding.

    Their prices stay in the pool: when unit price, quantity and line total
    disagree we do not know which is wrong, and dropping the row would be a
    guess presented as a correction. The count is disclosed instead.
    """
    cur.execute(
        """
        SELECT DISTINCT doc_pk_candidate FROM proc.bp_extraction_discrepancy
         WHERE status = 'open' AND doc_pk_candidate IS NOT NULL
        """
    )
    return {row[0] for row in cur.fetchall()}


def _count_suspect(
    point_ids: list[str], doc_by_point: dict[str, str], flagged: set[str]
) -> int:
    return sum(1 for pid in point_ids if doc_by_point.get(pid) in flagged)


def _to_points(pool_rows: list[dict[str, Any]]) -> list[BenchmarkPoint]:
    points = []
    for row in pool_rows:
        points.append(
            BenchmarkPoint(
                benchmark_point_id=row["point_id"],
                source="internal",
                item_name=_norm_item(row["item_description"]),
                uom=_norm_uom(row["unit_of_measure"]),
                currency=_norm_currency(row["currency"]),
                include=True,
                raw_unit_price=float(row["unit_price"]),
                source_weight=1.0,
                specification_score=_NEUTRAL_SCORE,
                location_cost_index=1.0,
                sla_score=_NEUTRAL_SCORE,
                historical_quantity=(
                    float(row["quantity"]) if row["quantity"] is not None else None
                ),
                index_value_at_price_date=1.0,
            )
        )
    return points


def benchmark_deal(
    cur,
    deal_id: str,
    settings: Optional[BenchmarkSettings] = None,
) -> dict[str, Any]:
    """Run the benchmark engine over every priced quote line of a deal."""
    settings = settings if settings is not None else BenchmarkSettings()
    quote_rows = load_quote_lines(cur, deal_id)
    # Load twice: once unscoped so the exclusion count is independently
    # verifiable, once scoped for the calculation the engine actually uses.
    full_pool = load_benchmark_pool(cur)
    scoped_pool = load_benchmark_pool(cur, exclude_deal_id=deal_id)
    own_excluded = _pool_delta(len(full_pool), len(scoped_pool))
    points = _to_points(scoped_pool)
    doc_by_point = {row["point_id"]: row["doc_id"] for row in scoped_pool}
    flagged = load_flagged_documents(cur)

    results = []
    for row in quote_rows:
        quote = QuoteLine(
            deal_id=deal_id,
            item_name=_norm_item(row["item_description"]),
            supplier_name="",
            quote_ref=str(row["quote_id"] or ""),
            category="",
            quantity=float(row["quantity"] or 0.0),
            uom=_norm_uom(row["unit_of_measure"]),
            currency=_norm_currency(row["currency"]),
            location=(row.get("region") or row.get("country") or ""),
            requested_spec_score=_NEUTRAL_SCORE,
            service_level="",
            requested_sla_score=_NEUTRAL_SCORE,
            index_id="",
            quoted_unit_price=float(row["unit_price"]),
        )
        # No lookup tables exist in the DB yet: empty tables make every miss
        # explicit via fallbacks_used instead of silently pricing with 1.0.
        result = compute_benchmark(quote, points, {}, {}, settings)
        payload = result.model_dump()
        payload["source_item_description"] = row["item_description"]
        payload["quote_line_id"] = row["quote_line_id"]
        payload["suspect_points"] = _count_suspect(
            result.matched_point_ids, doc_by_point, flagged)
        results.append(payload)

    gated = sum(1 for r in results if r["gated"])
    return {
        "deal_id": deal_id,
        "settings": settings.model_dump(),
        "pool_size": len(points),
        "lines": results,
        "line_count": len(results),
        "gated_count": gated,
        "computed_count": len(results) - gated,
        "own_documents_excluded": own_excluded,
        "disclosures": DISCLOSURES,
    }
