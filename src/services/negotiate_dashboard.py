"""Negotiate-page dashboard data, computed per deal from the final (_trgt)
procurement tables.

Every builder returns the exact JSON shape the frontend expects and uses SAFE
FALLBACKS (0 / "" / [] / None) where the underlying data does not exist yet —
e.g. priced negotiation rounds are only present once a negotiation workflow has
run, so offer history / round KPIs come back empty until then. Nothing is
fabricated; values are arithmetic over grounded document data.

Public entry point: ``build_negotiate_dashboard(deal_id, conn=None)``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

_CCY_SYMBOL = {"GBP": "£", "USD": "$", "EUR": "€", "JPY": "¥"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rows(cur, sql, params=()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _sym(currency: Optional[str]) -> str:
    return _CCY_SYMBOL.get((currency or "GBP").upper(), "")


def _money(v, currency: Optional[str] = "GBP") -> str:
    """Format a number as a currency string, abbreviating to K/M like the UI."""
    n = _f(v)
    if n is None:
        return ""
    s = _sym(currency)
    a = abs(n)
    if a >= 1_000_000:
        return f"{s}{n / 1_000_000:.1f}M"
    if a >= 100_000:
        return f"{s}{n / 1_000:.0f}K"
    if a >= 1_000:
        return f"{s}{n / 1_000:.1f}K"
    return f"{s}{n:,.0f}"


def _deal(cur, deal_id: str) -> Optional[dict]:
    rows = _rows(cur,
        "select deal_id, deal_name, supplier_id, supplier_name, deal_date, "
        "first_activity_date, last_activity_date, quote_count, po_count, "
        "invoice_count, quote_total, po_total, invoice_total, currency, "
        "price_variance_pct, cycle_days_quote_to_po, cycle_days_po_to_invoice, "
        "has_quote_anchor, orphaned "
        "from proc.bp_deal_overview where deal_id=%s", (deal_id,))
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# 1. summary  (deterministic, deal-specific; no LLM dependency in the endpoint)
# ---------------------------------------------------------------------------
def deal_summary_text(cur, deal_id: str, d: Optional[dict] = None) -> str:
    d = d or _deal(cur, deal_id)
    if not d:
        return ""
    ccy = d.get("currency") or "GBP"
    spend = d.get("invoice_total") or d.get("po_total") or d.get("quote_total")
    name = d.get("deal_name") or deal_id
    parts = [
        f"Deal '{name}' consolidates "
        f"{int(d.get('quote_count') or 0)} quote(s), "
        f"{int(d.get('po_count') or 0)} purchase order(s) and "
        f"{int(d.get('invoice_count') or 0)} invoice(s)"
    ]
    if d.get("supplier_name") or d.get("supplier_id"):
        parts.append(f"with supplier {d.get('supplier_name') or d.get('supplier_id')}")
    if spend is not None:
        parts.append(f"at a contract value of {_money(spend, ccy)}")
    pv = _f(d.get("price_variance_pct"))
    if pv is not None:
        parts.append(f"with a {pv:.1f}% price variance across the chain")
    return " ".join(parts) + "."


# ---------------------------------------------------------------------------
# 2. proposalSnapshot
# ---------------------------------------------------------------------------
def proposal_snapshot(cur, deal_id: str, d: Optional[dict] = None) -> list[dict]:
    d = d or _deal(cur, deal_id)
    if not d:
        return []
    ccy = d.get("currency") or "GBP"
    quote = _f(d.get("quote_total"))
    actual = _f(d.get("po_total")) or _f(d.get("invoice_total"))
    tvc = actual or quote or 0.0
    savings_pct = 0.0
    if quote and actual and quote > 0:
        savings_pct = round((quote - actual) / quote * 100, 1)
    # payment terms / contract term from the PO (then invoice)
    pterm = _rows(cur,
        "select payment_terms from proc.bp_purchase_order_trgt "
        "where deal_id=%s and payment_terms is not null limit 1", (deal_id,))
    payment_terms = pterm[0]["payment_terms"] if pterm else ""
    term = _contract_term(cur, deal_id)
    # The quote is the savings baseline. With no quote (orphaned chain) there is
    # nothing to measure against — surface that rather than a misleading 0%.
    savings_val = "awaiting quote" if not quote else f"{savings_pct:.0f}%"
    return [
        {"label": "tvc", "value": _money(tvc, ccy)},
        {"label": "annualRunRate", "value": _money(tvc, ccy)},
        {"label": "savingsVsBaseline", "value": savings_val},
        {"label": "paymentTerms", "value": payment_terms or ""},
        {"label": "Terms", "value": term or ""},
    ]


def _contract_term(cur, deal_id: str) -> str:
    try:
        rows = _rows(cur,
            "select c.contract_term, c.end_date from proc.bp_contracts c "
            "join proc.bp_purchase_order_trgt p on p.contract_id = c.contract_id "
            "where p.deal_id=%s limit 1", (deal_id,))
        if rows and rows[0].get("contract_term"):
            return str(rows[0]["contract_term"])
    except Exception:  # contracts table/columns optional
        pass
    return ""


# ---------------------------------------------------------------------------
# 3. version history of offers per round  (priced negotiation rounds)
# ---------------------------------------------------------------------------
_ROUND_LABELS = {1: "Initial offer", 2: "Revised offer", 3: "Further offer", 4: "Final offer"}


def offer_version_history(cur, deal_id: str, d: Optional[dict] = None) -> list[dict]:
    """Offer per negotiation round, newest first. Sourced from priced supplier
    responses for the deal's supplier; empty until a negotiation has run."""
    d = d or _deal(cur, deal_id)
    if not d or not d.get("supplier_id"):
        return []
    ccy = d.get("currency") or "GBP"
    try:
        rows = _rows(cur,
            "select round_number, max(price) price from proc.supplier_response "
            "where supplier_id=%s and price is not null "
            "group by round_number order by round_number desc", (d["supplier_id"],))
    except Exception:  # supplier_response table only exists once negotiations run
        return []
    out = []
    for r in rows:
        rn = int(r["round_number"]) if r.get("round_number") is not None else 0
        out.append({
            "round": f"Round {rn}",
            "label": _ROUND_LABELS.get(rn, "Offer"),
            "value": _money(r.get("price"), ccy),
        })
    return out


# ---------------------------------------------------------------------------
# 4. negotiation KPIs
# ---------------------------------------------------------------------------
def negotiation_kpis(cur, deal_id: str, d: Optional[dict] = None) -> list[dict]:
    d = d or _deal(cur, deal_id)
    if not d:
        return []
    ccy = d.get("currency") or "GBP"
    quote = _f(d.get("quote_total"))
    actual = _f(d.get("po_total")) or _f(d.get("invoice_total"))
    total_cost = actual or quote or 0.0
    savings = (quote - actual) if (quote and actual) else 0.0
    price_change_pct = round((actual - quote) / quote * 100, 1) if (quote and actual and quote > 0) else 0.0
    volume = _volume_total(cur, deal_id)
    closure = d.get("last_activity_date")
    closure_str = closure.strftime("%d %B %Y") if hasattr(closure, "strftime") else (str(closure) if closure else "")
    savings_val = "awaiting quote" if not quote else _money(savings, ccy)
    return [
        {"label": "Total Cost", "value": _money(total_cost, ccy), "positive": True},
        {"label": "Savings", "value": savings_val, "positive": savings >= 0},
        {"label": "Price Change", "value": f"{price_change_pct:.1f}%", "positive": price_change_pct <= 0},
        {"label": "Volume Change", "value": str(int(volume)) if volume is not None else "0",
         "positive": True},
        {"label": "closure Date", "value": closure_str},
        {"label": "Time per round", "value": _time_per_round(cur, d)},
    ]


def _volume_total(cur, deal_id: str) -> Optional[float]:
    rows = _rows(cur,
        "select coalesce(sum(quantity),0) v from proc.bp_po_line_items_trgt "
        "where deal_id=%s", (deal_id,))
    v = _f(rows[0]["v"]) if rows else None
    if not v:
        rows = _rows(cur,
            "select coalesce(sum(quantity),0) v from proc.bp_quote_line_items_trgt "
            "where deal_id=%s", (deal_id,))
        v = _f(rows[0]["v"]) if rows else None
    return v


def _time_per_round(cur, d: dict) -> str:
    try:
        rows = _rows(cur,
            "select count(distinct round_number) n from proc.supplier_response "
            "where supplier_id=%s", (d.get("supplier_id"),))
        n = int(rows[0]["n"]) if rows and rows[0].get("n") else 0
    except Exception:
        n = 0
    return str(n) if n else ""


# ---------------------------------------------------------------------------
# 5. negotiation strategy  (deterministic, driven by deal context)
# ---------------------------------------------------------------------------
def _advice_plays(deal_id: str) -> list[dict]:
    """Ranked, grounded plays. Imported lazily to avoid an import cycle."""
    from src.services.negotiation_advice import build_advice
    advice = build_advice(deal_id)
    return list((advice or {}).get("plays") or [])


def negotiation_strategy(cur, deal_id: str, d: Optional[dict] = None) -> list[dict]:
    d = d or _deal(cur, deal_id)
    if not d:
        return []
    quote = _f(d.get("quote_total"))
    actual = _f(d.get("po_total")) or _f(d.get("invoice_total"))
    saving = (quote - actual) if (quote and actual) else 0.0
    closure = d.get("last_activity_date")
    closure_str = closure.strftime("%d %B %Y") if hasattr(closure, "strftime") else ""
    rounds = 0
    try:
        rr = _rows(cur, "select count(distinct round_number) n from proc.supplier_response "
                        "where supplier_id=%s", (d.get("supplier_id"),))
        rounds = int(rr[0]["n"]) if rr and rr[0].get("n") else 0
    except Exception:
        rounds = 0
    high = (
        f"There have been {rounds} round(s) of negotiation with the supplier. "
        f"{'Savings of ' + _money(saving, d.get('currency')) + ' have been secured. ' if saving else ''}"
        f"{'Contract/closure date is ' + closure_str + '.' if closure_str else ''}"
    ).strip()
    # Standpoint as an indexed rate (100 = quote baseline). Only supplierRate
    # is derivable from this deal's own figures.
    #
    # Until 2026-09-05 this block also emitted `our_aim = supplier_rate - 3`
    # under the label "optimalPrice", and `walk_away = supplier_rate`. Neither
    # was computed from anything: the -3 was an arbitrary constant, and a
    # walk-away equal to the supplier's own rate asserts that we will never
    # walk away. With no quote or no actual, the literals 50/47/50 shipped to
    # the buyer unchanged. Deriving an optimum needs a should-cost or a
    # benchmark, and deriving a walk-away needs an authority limit; this
    # project has neither (see docs/negotiation-agent-state-audit.md 1.2, 1.6).
    # So the page now says it does not know.
    supplier_rate = None
    if quote and actual and quote > 0:
        supplier_rate = round(actual / quote * 50)
    our_aim = None
    walk_away = None
    unavailable_reason = (
        "An optimal price needs a should-cost or benchmark, and a walk-away "
        "needs an authority limit. Neither is available for this deal."
    )
    persona, key_driver, recommendation = _supplier_insights(cur, d)
    # leveragePoints and counterStrategy were two fixed strings shown to every
    # buyer on every deal — "Cost justification and benchmark variance" regardless
    # of whether this deal had a benchmark. They are replaced by plays the advisor
    # ranked for THIS deal, each carrying its state and the evidence behind it.
    # An advice failure costs the plays, not the dashboard: the rest of this
    # payload is computed independently.
    try:
        plays = _advice_plays(deal_id)
    except Exception:
        log.exception("advice plays unavailable for %s", deal_id)
        plays = []
    return [{
        "highLevelSummary": high,
        "currentStandpoint": {"supplierRate": supplier_rate, "ourAim": our_aim,
                              "walkAway": walk_away},
        "preferredOutcome": {"optimalPrice": our_aim, "targetRange": None,
                             "walkAway": walk_away,
                             "unavailableReason": unavailable_reason},
        "plays": plays,
        "supplierInsights": {"persona": persona, "keyDriver": key_driver,
                             "recommendation": recommendation},
    }]


def _supplier_insights(cur, d: dict):
    persona = "Analytical decision-maker"
    key_driver = "Revenue growth"
    recommendation = "Use structured data and phased trade-offs"
    try:
        rows = _rows(cur,
            "select is_preferred_supplier, risk_score from proc.bp_supplier where supplier_id=%s",
            (d.get("supplier_id"),))
        if rows:
            if rows[0].get("is_preferred_supplier"):
                persona = "Relationship-oriented partner"
            # risk_score is VARCHAR on a 0-100 scale (median 49.57), not 0-1.
            # The old `>= 0.6` matched 5000 of 5000 suppliers, so every deal
            # carried the same de-risking advice. Same bar the advice engine uses.
            from src.services.negotiation_advice.signals import RISK_ELEVATED
            risk = _f(rows[0].get("risk_score"))
            if risk is not None and risk >= RISK_ELEVATED:
                key_driver = "Risk and stability"
                recommendation = "De-risk with phased commitments and clear SLAs"
    except Exception:
        pass
    return persona, key_driver, recommendation


# ---------------------------------------------------------------------------
# 6. cost over time  (cumulative per doc type)
# ---------------------------------------------------------------------------
def cost_over_time(cur, deal_id: str) -> list[dict]:
    events = []  # (date, type, amount)
    for typ, table, datecol, amtcol in (
        ("Quote", "bp_quote_trgt", "quote_date", "total_amount"),
        ("PO", "bp_purchase_order_trgt", "order_date", "total_amount"),
        ("Invoice", "bp_invoice_trgt", "invoice_date", "invoice_amount"),
    ):
        for r in _rows(cur, f"select {datecol} dt, {amtcol} amt from proc.{table} "
                            f"where deal_id=%s and {datecol} is not null order by {datecol}", (deal_id,)):
            amt = _f(r["amt"])
            if r["dt"] is not None and amt is not None:
                events.append((r["dt"], typ, amt))
    if not events:
        return []
    events.sort(key=lambda e: e[0])
    cum = {"PO": 0.0, "Invoice": 0.0, "Quote": 0.0}
    out = []
    for dt, typ, amt in events:
        cum[typ] += amt
        # Quote, PO and Invoice for a three-way-matched deal all represent the
        # SAME spend, so summing them triple-counts. "Overall" is the actual
        # cost consumed: prefer the most concrete stream (invoiced), falling
        # back to committed (PO) then proposed (Quote).
        overall = cum["Invoice"] or cum["PO"] or cum["Quote"]
        out.append({
            "time": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
            "PO": round(cum["PO"], 2), "Invoice": round(cum["Invoice"], 2),
            "Quote": round(cum["Quote"], 2),
            "Overall": round(overall, 2),
        })
    return out


# ---------------------------------------------------------------------------
# 7. volume trend  (per month: total quantity + avg unit price)
# ---------------------------------------------------------------------------
def volume_trend(cur, deal_id: str) -> list[dict]:
    rows = _rows(cur,
        "select to_char(date_trunc('month', p.order_date),'Mon') mon, "
        "date_trunc('month', p.order_date) m, "
        "coalesce(sum(li.quantity),0) vol, avg(li.unit_price) aup "
        "from proc.bp_po_line_items_trgt li "
        "join proc.bp_purchase_order_trgt p on p.po_id=li.po_id "
        "where p.deal_id=%s and p.order_date is not null "
        "group by 1,2 order by 2", (deal_id,))
    if not rows:
        # Quote-only deals have no POs yet — fall back to the proposed volume
        # from quotes so the chart reflects the deal instead of rendering blank.
        # Mirrors the Volume KPI fallback in _volume_total.
        rows = _rows(cur,
            "select to_char(date_trunc('month', q.quote_date),'Mon') mon, "
            "date_trunc('month', q.quote_date) m, "
            "coalesce(sum(li.quantity),0) vol, avg(li.unit_price) aup "
            "from proc.bp_quote_line_items_trgt li "
            "join proc.bp_quote_trgt q on q.quote_id=li.quote_id "
            "where q.deal_id=%s and q.quote_date is not null "
            "group by 1,2 order by 2", (deal_id,))
    return [{"month": r["mon"], "Volume": round(_f(r["vol"]) or 0, 2),
             "AvgUnitPrice": round(_f(r["aup"]) or 0, 2)} for r in rows]


# ---------------------------------------------------------------------------
# 8. proposal summary  (baseline quote unit vs current PO/invoice unit per item)
# ---------------------------------------------------------------------------
def proposal_summary(cur, deal_id: str, d: Optional[dict] = None) -> list[dict]:
    d = d or _deal(cur, deal_id)
    ccy = (d or {}).get("currency") or "GBP"
    base = {r["item"]: _f(r["u"]) for r in _rows(cur,
        "select coalesce(item_description, item_id) item, avg(unit_price) u "
        "from proc.bp_quote_line_items_trgt where deal_id=%s and unit_price is not null "
        "group by 1", (deal_id,))}
    curr = {r["item"]: _f(r["u"]) for r in _rows(cur,
        "select coalesce(item_description, item_id) item, avg(unit_price) u "
        "from proc.bp_po_line_items_trgt where deal_id=%s and unit_price is not null "
        "group by 1", (deal_id,))}
    out = []
    for item in sorted(set(base) | set(curr)):
        b, c = base.get(item), curr.get(item)
        diff = (c - b) if (b is not None and c is not None) else None
        out.append({
            "item": item,
            "baselineUnit": _money(b, ccy) if b is not None else "",
            "currentUnit": _money(c, ccy) if c is not None else "",
            "unit": (("-" if diff < 0 else "+") + _money(abs(diff), ccy)) if diff is not None else "",
        })
    return out


# ---------------------------------------------------------------------------
# 9. demand vs volume  (volume real; demand forecast not tracked -> null)
# ---------------------------------------------------------------------------
def demand_vs_volume(cur, deal_id: str) -> list[dict]:
    rows = _rows(cur,
        "select to_char(date_trunc('month', p.order_date),'Mon') mon, "
        "date_trunc('month', p.order_date) m, coalesce(sum(li.quantity),0) vol "
        "from proc.bp_po_line_items_trgt li "
        "join proc.bp_purchase_order_trgt p on p.po_id=li.po_id "
        "where p.deal_id=%s and p.order_date is not null group by 1,2 order by 2", (deal_id,))
    if not rows:
        # Quote-only deals: fall back to proposed volume from quotes (see volume_trend).
        rows = _rows(cur,
            "select to_char(date_trunc('month', q.quote_date),'Mon') mon, "
            "date_trunc('month', q.quote_date) m, coalesce(sum(li.quantity),0) vol "
            "from proc.bp_quote_line_items_trgt li "
            "join proc.bp_quote_trgt q on q.quote_id=li.quote_id "
            "where q.deal_id=%s and q.quote_date is not null group by 1,2 order by 2", (deal_id,))
    return [{"month": r["mon"], "demand": None, "volume": round(_f(r["vol"]) or 0, 2)} for r in rows]


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def build_negotiate_dashboard(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Assemble the full negotiate-page payload for a deal, or None if unknown."""
    if conn is None:
        with get_conn() as own:
            return _build(own.cursor(), deal_id)
    return _build(conn.cursor(), deal_id)


def _build(cur, deal_id: str) -> Optional[dict]:
    d = _deal(cur, deal_id)
    if not d:
        return None
    return {
        "deal_id": deal_id,
        "deal_name": d.get("deal_name"),
        "orphaned": bool(d.get("orphaned")),
        "hasQuoteAnchor": bool(d.get("has_quote_anchor")),
        "summary": deal_summary_text(cur, deal_id, d),
        "proposalSnapshot": proposal_snapshot(cur, deal_id, d),
        "versionHistory": offer_version_history(cur, deal_id, d),
        "negotiationData": negotiation_kpis(cur, deal_id, d),
        "negotiationStrategy": negotiation_strategy(cur, deal_id, d),
        "costOverTime": cost_over_time(cur, deal_id),
        "volumeTrend": volume_trend(cur, deal_id),
        "proposalSummary": proposal_summary(cur, deal_id, d),
        "demandVsVolumeData": demand_vs_volume(cur, deal_id),
    }
