"""Deterministic per-deal analytics + summary sync.

compute_deal_metrics() turns a deal's final (_trgt) documents into the metrics
row backing the UI grid — no LLM, no fabrication. sync_deal_summaries() (Task 3)
persists those rows and the AgentNick narrative for every Deal_Linked deal.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn
from src.services.deal_summary import gather_deal_context

log = logging.getLogger(__name__)


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _first_present(docs: list[dict], key: str):
    for d in docs:
        val = d.get(key)
        if val not in (None, ""):
            return val
    return None


def _doc_total(docs: list[dict]) -> Optional[float]:
    """Sum of header total_amount across docs of one type; None if none present."""
    vals = [_num(d.get("total_amount")) for d in docs]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _sum_qty(docs: list[dict]) -> Optional[float]:
    total = 0.0
    seen = False
    for d in docs:
        for li in d.get("line_items") or []:
            q = _num(li.get("quantity"))
            if q is not None:
                total += q
                seen = True
    return total if seen else None


def _weighted_unit_price(docs: list[dict]) -> Optional[float]:
    """Total line value / total qty across a doc type's line items."""
    val = 0.0
    qty = 0.0
    seen = False
    for d in docs:
        for li in d.get("line_items") or []:
            q = _num(li.get("quantity"))
            up = _num(li.get("unit_price"))
            if q is not None and up is not None:
                val += q * up
                qty += q
                seen = True
    if not seen or qty == 0:
        return None
    return val / qty


def _pct_change(new: Optional[float], base: Optional[float]) -> Optional[float]:
    if new is None or base is None or base == 0:
        return None
    return round((new - base) / base * 100.0, 2)


def _deal_category(cur, deal_id: str) -> Optional[str]:
    cur.execute(
        "select category from proc.process_monitor "
        "where deal_id = %s and category is not null limit 1",
        (deal_id,))
    row = cur.fetchone()
    return row[0] if row else None


def _items_from(docs: list[dict]) -> list[dict]:
    items: list[dict] = []
    for d in docs:
        for li in d.get("line_items") or []:
            name = li.get("item_description")
            if name:
                items.append({"name": name,
                              "qty": _num(li.get("quantity")),
                              "unit_price": _num(li.get("unit_price"))})
    return items


def _compute(ctx: dict, cur) -> dict:
    docs = ctx["documents"]
    inv, po, quote = docs["invoices"], docs["purchase_orders"], docs["quotes"]

    supplier = (_first_present(inv, "supplier_name")
                or _first_present(po, "supplier_name")
                or _first_present(quote, "supplier_name"))

    # deal value / currency: prefer invoice, then PO, then quote
    deal_value = _doc_total(inv)
    currency = _first_present(inv, "currency")
    if deal_value is None:
        deal_value, currency = _doc_total(po), _first_present(po, "currency")
    if deal_value is None:
        deal_value, currency = _doc_total(quote), _first_present(quote, "currency")

    # volume: prefer invoice line qty, then PO, then quote
    volume = _sum_qty(inv) or _sum_qty(po) or _sum_qty(quote)
    unit_price = (deal_value / volume) if (deal_value is not None and volume) else None

    inv_unit = _weighted_unit_price(inv)
    quote_unit = _weighted_unit_price(quote) or _weighted_unit_price(po)
    price_change_pct = _pct_change(inv_unit, quote_unit)

    inv_vol = _sum_qty(inv)
    quote_vol = _sum_qty(quote) or _sum_qty(po)
    volume_change_pct = _pct_change(inv_vol, quote_vol)

    # efficiency = realized savings = (quoted unit - invoiced unit) * invoiced volume
    efficiency_score = None
    if inv_unit is not None and quote_unit is not None and inv_vol is not None:
        efficiency_score = round((quote_unit - inv_unit) * inv_vol, 2)

    items = _items_from(inv) or _items_from(quote) or _items_from(po)

    return {
        "deal_id": ctx["deal_id"],
        "deal_name": ctx.get("deal_name"),
        "supplier": supplier,
        "category": _deal_category(cur, ctx["deal_id"]),
        "deal_value": round(deal_value, 2) if deal_value is not None else None,
        "currency": currency,
        "volume": volume,
        "unit_price": round(unit_price, 4) if unit_price is not None else None,
        "price_change_pct": price_change_pct,
        "volume_change_pct": volume_change_pct,
        "efficiency_score": efficiency_score,
        "items": items,
        "item_count": len(items),
        "data_snapshot": {
            "invoice_total": _doc_total(inv), "po_total": _doc_total(po),
            "quote_total": _doc_total(quote), "invoice_unit": inv_unit,
            "quote_unit": quote_unit, "invoice_volume": inv_vol,
            "quote_volume": quote_vol,
        },
    }


def compute_deal_metrics(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Deterministic metrics for a deal. None if no final records exist."""
    if conn is not None:
        ctx = gather_deal_context(deal_id, conn=conn)
        if ctx is None:
            return None
        cur = conn.cursor() if hasattr(conn, 'cursor') else None
        return _compute(ctx, cur)
    with get_conn() as own:
        ctx = gather_deal_context(deal_id, conn=own)
        if ctx is None:
            return None
        return _compute(ctx, own.cursor())
