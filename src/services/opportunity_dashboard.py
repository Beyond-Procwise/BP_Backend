"""Opportunities-page dashboard, aggregated from proc.bp_opportunity.

Returns the exact JSON shapes the frontend expects with safe fallbacks. Stage
lifecycle: identified -> negotiation -> agreed -> realised, plus closed /
rejected. "Open/in-flight" = negotiation|agreed; "closed" = realised|closed|
rejected. Money is abbreviated (£/K/M) to match the design.

Public entry point: ``build_opportunities_dashboard(conn=None)``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

_OPEN = ("identified", "negotiation", "agreed")
_INFLIGHT = ("negotiation", "agreed")
_CLOSED = ("realised", "closed", "rejected")


def _rows(cur, sql, params=()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _f(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _money(v) -> str:
    n = _f(v)
    a = abs(n)
    if a >= 1_000_000:
        return f"£{n / 1_000_000:.1f}M"
    if a >= 100_000:
        return f"£{n / 1_000:.0f}k"
    if a >= 1_000:
        return f"£{n / 1_000:.1f}k"
    return f"£{n:,.0f}"


def _pct_change(curr: float, prev: float) -> str:
    if prev == 0:
        return "+0%" if curr == 0 else "+100%"
    pct = round((curr - prev) / abs(prev) * 100)
    return f"{'+' if pct >= 0 else ''}{pct}%"


# ---------------------------------------------------------------------------
# 1. headline KPIs
# ---------------------------------------------------------------------------
def opportunities_data(cur) -> dict:
    agg = _rows(cur,
        "select "
        " count(*) total, "
        " count(*) filter (where stage = any(%s)) identified, "
        " count(*) filter (where stage = any(%s)) closed, "
        " count(*) filter (where stage = any(%s)) in_flight, "
        " coalesce(sum(financial_impact_gbp),0) potential, "
        " coalesce(sum(realised_savings_gbp) filter (where stage='realised'),0) realised, "
        " count(distinct category_id) filter (where category_id is not null) cat_impact "
        "from proc.bp_opportunity",
        (list(_OPEN), list(_CLOSED), list(_INFLIGHT)))
    a = agg[0] if agg else {}
    # period-over-period change: this month vs previous month (by detected_on)
    cur_prev = _rows(cur,
        "select "
        " count(*) filter (where date_trunc('month',detected_on)=date_trunc('month',now())) cur_n, "
        " count(*) filter (where date_trunc('month',detected_on)=date_trunc('month',now()-interval '1 month')) prev_n, "
        " coalesce(sum(financial_impact_gbp) filter (where date_trunc('month',detected_on)=date_trunc('month',now())),0) cur_p, "
        " coalesce(sum(financial_impact_gbp) filter (where date_trunc('month',detected_on)=date_trunc('month',now()-interval '1 month')),0) prev_p, "
        " coalesce(sum(realised_savings_gbp) filter (where stage='realised' and date_trunc('month',stage_updated_at)=date_trunc('month',now())),0) cur_r, "
        " coalesce(sum(realised_savings_gbp) filter (where stage='realised' and date_trunc('month',stage_updated_at)=date_trunc('month',now()-interval '1 month')),0) prev_r, "
        " count(*) filter (where stage=any(%s) and date_trunc('month',stage_updated_at)=date_trunc('month',now())) cur_if, "
        " count(*) filter (where stage=any(%s) and date_trunc('month',stage_updated_at)=date_trunc('month',now()-interval '1 month')) prev_if "
        "from proc.bp_opportunity", (list(_INFLIGHT), list(_INFLIGHT)))
    c = cur_prev[0] if cur_prev else {}
    return {
        "totalOpportunity": int(_f(a.get("total"))),
        "opportunitiesChange": _pct_change(_f(c.get("cur_n")), _f(c.get("prev_n"))),
        "identified": int(_f(a.get("identified"))),
        "closed": int(_f(a.get("closed"))),
        "potential": _money(a.get("potential")),
        "potentialChange": _pct_change(_f(c.get("cur_p")), _f(c.get("prev_p"))),
        "realised": _money(a.get("realised")),
        "realisedChange": _pct_change(_f(c.get("cur_r")), _f(c.get("prev_r"))),
        "categoryImpact": int(_f(a.get("cat_impact"))),
        "catImpChange": _pct_change(0, 0),
        "inFlight": int(_f(a.get("in_flight"))),
        "inFlightChange": _pct_change(_f(c.get("cur_if")), _f(c.get("prev_if"))),
    }


# ---------------------------------------------------------------------------
# 2. savings pipeline funnel
# ---------------------------------------------------------------------------
def savings_pipeline(cur) -> list[dict]:
    a = (_rows(cur,
        "select "
        " coalesce(sum(financial_impact_gbp),0) identified, "
        " coalesce(sum(financial_impact_gbp) filter (where stage=any(%s)),0) negotiation, "
        " coalesce(sum(financial_impact_gbp) filter (where stage in ('agreed','realised')),0) agreed, "
        " coalesce(sum(realised_savings_gbp) filter (where stage='realised'),0) realised "
        "from proc.bp_opportunity",
        (["negotiation", "agreed", "realised"],)) or [{}])[0]
    return [
        {"label": "opportunities Identified", "value": _money(a.get("identified"))},
        {"label": "negotiation Started", "value": _money(a.get("negotiation"))},
        {"label": "savings Agreed", "value": _money(a.get("agreed"))},
        {"label": "savings Realized", "value": _money(a.get("realised"))},
    ]


# ---------------------------------------------------------------------------
# 3. savings identified vs completed by month
# ---------------------------------------------------------------------------
def savings_identified_vs_completed(cur) -> list[dict]:
    ident = {r["mon"]: (_f(r["v"]), r["m"]) for r in _rows(cur,
        "select to_char(date_trunc('month',detected_on),'Mon') mon, date_trunc('month',detected_on) m, "
        "coalesce(sum(financial_impact_gbp),0) v from proc.bp_opportunity "
        "where detected_on is not null group by 1,2")}
    comp = {r["mon"]: _f(r["v"]) for r in _rows(cur,
        "select to_char(date_trunc('month',stage_updated_at),'Mon') mon, "
        "coalesce(sum(realised_savings_gbp),0) v from proc.bp_opportunity "
        "where stage='realised' and stage_updated_at is not null group by 1")}
    out = [{"month": mon, "identified": round(v), "completed": round(comp.get(mon, 0.0)), "_m": m}
           for mon, (v, m) in ident.items()]
    out.sort(key=lambda x: x["_m"])
    for x in out:
        x.pop("_m", None)
    return out


# ---------------------------------------------------------------------------
# 4. opportunity trends by month (count)
# ---------------------------------------------------------------------------
def opportunity_trends(cur) -> list[dict]:
    rows = _rows(cur,
        "select to_char(date_trunc('month',detected_on),'Mon') mon, date_trunc('month',detected_on) m, "
        "count(*) n from proc.bp_opportunity where detected_on is not null group by 1,2 order by 2")
    return [{"month": r["mon"], "opportunities": int(_f(r["n"]))} for r in rows]


# ---------------------------------------------------------------------------
# 5. detailed opportunities list
# ---------------------------------------------------------------------------
def detailed_opportunities(cur, limit: int = 100) -> list[dict]:
    # orphaned = the opportunity's deal has no anchoring quote (quote-anchored model).
    rows = _rows(cur,
        "select o.opportunity_id, o.detected_on, o.detector_type, o.category_id, "
        "o.supplier_name, o.supplier_id, o.item_description, o.item_id, "
        "o.financial_impact_gbp, o.stage, o.quote_id, o.deal_id, "
        "(o.deal_id is not null and o.deal_id<>'' "
        " and not exists(select 1 from proc.bp_quote_trgt q where q.deal_id=o.deal_id)) as orphaned "
        "from proc.bp_opportunity o "
        "order by o.financial_impact_gbp desc nulls last, o.detected_on desc nulls last "
        "limit %s", (limit,))
    out = []
    for r in rows:
        det = r.get("detected_on")
        out.append({
            "opportunityId": r.get("opportunity_id"),
            "date": det.strftime("%Y-%m-%d") if hasattr(det, "strftime") else (str(det)[:10] if det else ""),
            "type": r.get("detector_type") or "",
            "category": r.get("category_id") or "Uncategorised",
            "supplier": r.get("supplier_name") or r.get("supplier_id") or "",
            "opportunity": r.get("item_description") or r.get("item_id") or (r.get("detector_type") or ""),
            "potentialSaving": _money(r.get("financial_impact_gbp")),
            "stage": r.get("stage"),
            "quoteId": r.get("quote_id"),
            "orphaned": bool(r.get("orphaned")),
        })
    return out


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def build_opportunities_dashboard(conn: Any = None, limit: int = 100) -> dict:
    if conn is None:
        with get_conn() as own:
            return _build(own.cursor(), limit)
    return _build(conn.cursor(), limit)


def _build(cur, limit: int) -> dict:
    return {
        "opportunitiesData": opportunities_data(cur),
        "savingsPipeline": savings_pipeline(cur),
        "savingsIdentifiedVsCompleted": savings_identified_vs_completed(cur),
        "opportunityTrends": opportunity_trends(cur),
        "detailedOpportunities": detailed_opportunities(cur, limit),
    }
