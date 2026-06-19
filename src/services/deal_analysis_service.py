"""Deterministic per-deal analytics + summary sync.

compute_deal_metrics() turns a deal's final (_trgt) documents into the metrics
row backing the UI grid — no LLM, no fabrication. sync_deal_summaries() (Task 3)
persists those rows and the AgentNick narrative for every Deal_Linked deal.
"""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    volume = _sum_qty(inv)
    if volume is None:
        volume = _sum_qty(po)
    if volume is None:
        volume = _sum_qty(quote)
    unit_price = (deal_value / volume) if (deal_value is not None and volume) else None

    inv_unit = _weighted_unit_price(inv)
    quote_unit = _weighted_unit_price(quote)
    if quote_unit is None:
        quote_unit = _weighted_unit_price(po)
    price_change_pct = _pct_change(inv_unit, quote_unit)

    inv_vol = _sum_qty(inv)
    quote_vol = _sum_qty(quote)
    if quote_vol is None:
        quote_vol = _sum_qty(po)
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
        "category": _deal_category(cur, ctx["deal_id"]) if cur is not None else None,
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


_NARRATIVE_PERSONA = "analysis"
_NARRATIVE_SOURCE = "deal_analysis_service"


def upsert_analysis_row(conn: Any, metrics: dict, narrative_summary_id: Optional[str],
                        model: Optional[str], summary_text: Optional[str] = None) -> str:
    """Replace the deal's row: delete any prior row(s) for the deal, then insert
    the new one. The table holds exactly ONE row per deal (no is_current history).

    The DELETE and the INSERT are committed together so a reader never sees the
    deal with zero rows mid-update. Because the live connection is autocommit=True
    we toggle it around the two statements, with a safe fallback for test fakes
    that lack the attribute.
    """
    analysis_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)

    prev_autocommit = getattr(conn, "autocommit", None)
    if prev_autocommit:
        try:
            conn.autocommit = False
        except Exception:
            prev_autocommit = None

    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM proc.bp_analysis_summary WHERE deal_id = %s",
            (metrics["deal_id"],))
        cur.execute(
            "INSERT INTO proc.bp_analysis_summary "
            "(analysis_id, deal_id, deal_name, supplier, category, deal_value, currency, "
            " volume, unit_price, price_change_pct, volume_change_pct, efficiency_score, "
            " items, item_count, narrative_summary_id, summary, data_snapshot, model, "
            " is_current, generated_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (analysis_id, metrics["deal_id"], metrics.get("deal_name"),
             metrics.get("supplier"), metrics.get("category"), metrics.get("deal_value"),
             metrics.get("currency"), metrics.get("volume"), metrics.get("unit_price"),
             metrics.get("price_change_pct"), metrics.get("volume_change_pct"),
             metrics.get("efficiency_score"),
             json.dumps(metrics.get("items"), default=str),
             metrics.get("item_count"), narrative_summary_id, summary_text,
             json.dumps(metrics.get("data_snapshot"), default=str),
             model, True, generated_at))
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if prev_autocommit:
            try:
                conn.autocommit = True
            except Exception:
                pass

    return analysis_id


def generate_for_deal(deal_id: str, conn: Any) -> dict:
    """Compute metrics, generate+store the AgentNick narrative, upsert the row."""
    from src.services.deal_summary import summarize_deal, _SUMMARY_MODEL
    from src.services.summary_agent import _store_summary

    metrics = compute_deal_metrics(deal_id, conn=conn)
    if metrics is None:
        return {"deal_id": deal_id, "status": "no_records"}

    narrative_id = None
    narrative_text = None
    try:
        narr = summarize_deal(deal_id, conn=conn)
        if narr and narr.get("summary"):
            narrative_text = narr["summary"]
            stored = _store_summary(
                conn, persona=_NARRATIVE_PERSONA, persona_source=_NARRATIVE_SOURCE,
                scope="deal", deal_id=deal_id, summary=narrative_text,
                data_snapshot=metrics.get("data_snapshot"),
                sources=narr.get("sources"), model=_SUMMARY_MODEL, is_current=True)
            narrative_id = stored["summary_id"]
    except Exception as exc:  # narrative is best-effort; metrics still persist
        log.warning("narrative generation failed for %s: %s", deal_id, exc)

    # Store the narrative TEXT directly on the analysis row (authoritative home),
    # alongside the bp_summary FK for the existing summary ecosystem.
    upsert_analysis_row(conn, metrics, narrative_id,
                        _SUMMARY_MODEL if narrative_id else None,
                        summary_text=narrative_text)
    return {"deal_id": deal_id, "status": "ok", "narrative_summary_id": narrative_id}


def _linked_deal_ids_needing_summary(cur) -> list[str]:
    """Deals at Deal_Linked status with no current bp_analysis_summary row."""
    cur.execute(
        "select distinct deal_id from proc.process_monitor pm "
        "where pm.status = 'Deal_Linked' and coalesce(pm.deal_id,'') <> '' "
        "and not exists (select 1 from proc.bp_analysis_summary a "
        "                where a.deal_id = pm.deal_id and a.is_current)")
    return [r[0] for r in cur.fetchall()]


def sync_deal_summaries(conn: Any = None, deal_ids: Optional[list[str]] = None,
                        max_workers: int = 2) -> dict:
    """Generate metrics + narrative for every linked deal missing a current summary.

    Each deal is processed on its own connection so failures stay isolated and
    work runs concurrently. Safe to call repeatedly (idempotent).
    """
    if conn is not None:
        ids = deal_ids if deal_ids is not None else _linked_deal_ids_needing_summary(conn.cursor())
    else:
        with get_conn() as own:
            ids = deal_ids if deal_ids is not None else _linked_deal_ids_needing_summary(own.cursor())

    processed = failed = 0
    done: list[str] = []

    def _work(deal_id: str):
        # Each worker uses an independent connection (psycopg connections are
        # not thread-safe to share). When a conn was passed in we still open a
        # fresh one per deal to keep failures from poisoning a shared txn.
        with get_conn() as wc:
            return generate_for_deal(deal_id, wc)

    if not ids:
        return {"processed": 0, "skipped": 0, "failed": 0, "deal_ids": []}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_work, d): d for d in ids}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                fut.result()
                processed += 1
                done.append(d)
            except Exception as exc:
                failed += 1
                log.warning("summary sync failed for deal %s: %s", d, exc)

    return {"processed": processed, "skipped": 0, "failed": failed, "deal_ids": done}


_CCY = {"GBP": "£", "USD": "$", "EUR": "€"}


def _money(value, currency) -> str:
    if value is None:
        return "–"
    sym = _CCY.get((currency or "").upper(), (currency + " ") if currency else "")
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"{sym}{v / 1_000_000:.1f}M".replace(".0M", "M")
    if abs(v) >= 1_000:
        return f"{sym}{round(v / 1_000)}K"
    return f"{sym}{v:,.2f}"


def _signed_pct(value) -> str:
    if value is None:
        return "–"
    return f"{'+' if value >= 0 else ''}{value:g}%"


def to_ui_row(row: dict) -> dict:
    """Map a bp_analysis_summary row to the exact shape AnalysisSummary.jsx wants.

    Every value is a string (the UI search filter lowercases each field), NULL
    numerics render as the en-dash, and items is a comma-joined product string.
    """
    cur = row.get("currency")
    items = row.get("items")
    if isinstance(items, str):
        try:
            import json as _json
            items = _json.loads(items)
        except Exception:
            items = None
    if isinstance(items, list):
        names = [i.get("name") for i in items if isinstance(i, dict) and i.get("name")]
        items_str = ", ".join(names) if names else "–"
    else:
        items_str = "–"
    unit = row.get("unit_price")
    return {
        "id": row.get("deal_id") or "–",
        "supplier": row.get("supplier") or "–",
        "category": row.get("category") or "–",
        "value": _money(row.get("deal_value"), cur),
        "volume": f"{float(row['volume']):,.0f}" if row.get("volume") is not None else "–",
        "unitPrice": _money(unit, cur) if unit is not None and float(unit) >= 1000
                     else (f"{_CCY.get((cur or '').upper(), (cur + ' ') if cur else '')}{float(unit):,.2f}"
                           if unit is not None else "–"),
        "priceChange": _signed_pct(row.get("price_change_pct")),
        "volumeChange": _signed_pct(row.get("volume_change_pct")),
        "efficiency": f"{float(row['efficiency_score']):g}" if row.get("efficiency_score") is not None else "–",
        "items": items_str,
    }
