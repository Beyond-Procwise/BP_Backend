"""Deal facts and the two scorer dicts, computed from grounded data only.

There is no category dimension in this database (proc.bp_category holds 0 rows
and no _trgt table has a category column), so supply-market competitiveness is
measured by item-level supplier overlap instead: how many distinct suppliers have
quoted the same item descriptions as this deal.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

_ALT_SQL = """
select count(distinct q2.supplier_id) as n
from proc.bp_quote_line_items_trgt mine
join proc.bp_quote_line_items_trgt theirs
  on lower(trim(theirs.item_description)) = lower(trim(mine.item_description))
join proc.bp_quote_trgt q2 on q2.quote_id = theirs.quote_id
where mine.deal_id = %s
  and mine.item_description is not null
  and length(trim(mine.item_description)) > 3
  and q2.supplier_id is not null
"""


def _rows(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _f(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def gather_signals(cur, deal_id: str) -> Optional[dict]:
    """Grounded facts for one deal, or None when the deal is unknown."""
    deal_rows = _rows(
        cur,
        "select deal_id, supplier_id, supplier_name, currency, quote_count, "
        "quote_total, po_total, invoice_total, price_variance_pct "
        "from proc.bp_deal_overview where deal_id=%s",
        (deal_id,),
    )
    if not deal_rows:
        return None
    d = deal_rows[0]

    alt: Optional[int] = None
    try:
        alt_rows = _rows(cur, _ALT_SQL, (deal_id,))
        if alt_rows:
            raw = alt_rows[0].get("n")
            alt = int(raw) if raw not in (None, 0) else None
    except Exception:
        log.debug("alternative-supplier count failed for %s", deal_id,
                  exc_info=True)

    risk: Optional[float] = None
    preferred: Optional[bool] = None
    if d.get("supplier_id"):
        try:
            sup = _rows(cur, "select risk_score, is_preferred_supplier "
                             "from proc.bp_supplier where supplier_id=%s",
                        (d["supplier_id"],))
            if sup:
                risk = _f(sup[0].get("risk_score"))
                raw_pref = sup[0].get("is_preferred_supplier")
                preferred = bool(raw_pref) if raw_pref is not None else None
        except Exception:
            log.debug("supplier lookup failed for %s", d.get("supplier_id"),
                      exc_info=True)

    invoice_total = _f(d.get("invoice_total"))
    po_total = _f(d.get("po_total"))
    quote_total = _f(d.get("quote_total"))
    return {
        "deal_id": d.get("deal_id"),
        "supplier_id": d.get("supplier_id"),
        "supplier_name": d.get("supplier_name"),
        "currency": d.get("currency"),
        "deal_value": invoice_total or po_total or quote_total,
        "invoice_total": invoice_total,
        "po_total": po_total,
        "quote_supplier_count": int(d.get("quote_count") or 0) or None,
        "alternative_supplier_count": alt,
        "risk_score": risk,
        "is_preferred": preferred,
        "price_variance_pct": _f(d.get("price_variance_pct")),
    }


def supplier_performance_dict(signals: dict) -> dict:
    """Only keys _score_supplier_performance actually reads, and only when known."""
    out: dict = {}
    on_time = signals.get("on_time_ratio")
    if on_time is not None:
        out["on_time_delivery"] = on_time
    return out


RISK_ELEVATED = 60.0        # risk_score is a 0-100 scale here, median 49.57
THIN_MARKET_ALTERNATIVES = 93   # the per-deal median; below it the market is thin


def market_context_dict(signals: dict) -> dict:
    """Only keys _score_market_context reads, and only when known.

    An uncomputable signal is omitted: the scorer returns (0.0, []) on an empty
    dict, which is the honest "no signal" outcome. Defaulting would invent a nudge.

    Both thresholds are on measured scales. `risk_score` is stored as VARCHAR on
    a 0-100 scale (min 5.00, median 49.57, max 94.94) — comparing it against 0.6
    matches 5000 of 5000 suppliers, i.e. always. `alternative_supplier_count` is
    a per-deal union whose median is 93, so a "thin market" test of <= 2 would
    never fire.

    The scorer treats "high", "elevated" and "tight" identically, so there is one
    tier and one string rather than a false distinction.
    """
    out: dict = {}
    alt = signals.get("alternative_supplier_count")
    risk = signals.get("risk_score")
    if alt is not None and alt < THIN_MARKET_ALTERNATIVES:
        out["supply_risk"] = "elevated"
    elif risk is not None and risk >= RISK_ELEVATED:
        out["supply_risk"] = "elevated"
    return out
