"""Presentation data: synthetic, deterministic, internally consistent, and never real.

One fictional organisation, fictional suppliers with plainly synthetic names, no real customer,
supplier or person. Every figure is derived from one table of synthetic facts, so groups sum to
their totals and period slices tie. It lives server-side behind the same contract as the live
provider; it is never shipped to the browser and never a fallback for live data.
"""
from __future__ import annotations

import datetime as dt
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from . import registry as R
from .live import Row

ORG = "Fictional Holdings (synthetic organisation)"
START_YEAR = 2024
N_SUPPLIERS = 12
CATEGORIES = ["Category A", "Category B", "Category C", "Category D", "Category E", "Category F"]
BANDS = ["Strategic", "Leverage", "Routine", "Tail"]
BUYERS = ["Synthetic Buyer 1", "Synthetic Buyer 2", "Synthetic Buyer 3"]
REGIONS = ["Region North", "Region South", "Region East"]
RISK_DIMS = ["Financial", "Operational", "Compliance", "Cyber", "Geographic", "Reputational"]


def _sup(i: int) -> Tuple[str, str]:
    return f"SYN-SUP-{i:02d}", f"Synthetic Supplier {i:02d}"


def _months(today: dt.date) -> List[dt.date]:
    out, y, m = [], START_YEAR, 1
    while (y, m) <= (today.year, today.month):
        out.append(dt.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@lru_cache(maxsize=8)
def _facts(today: dt.date) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for mi, month in enumerate(_months(today)):
        for si in range(1, N_SUPPLIERS + 1):
            sid, sname = _sup(si)
            cat = CATEGORIES[(si - 1) % len(CATEGORIES)]
            rnd = random.Random(f"{sid}-{month.isoformat()}")
            band = BANDS[min(3, (si - 1) // 3)]
            spend = round((60000 / si ** 0.6) * (1 + 0.012 * mi) * (0.85 + rnd.random() * 0.3))
            off = round(spend * (0.04 + 0.01 * (si % 5)) * (0.8 + rnd.random() * 0.4))
            deals = 2 + rnd.randrange(4)
            assessed = deals if rnd.random() > 0.15 else 0
            rows.append({
                "month": month, "supplier": sid, "supplier_label": sname, "category": cat, "tail_band": band,
                "buyer": BUYERS[si % len(BUYERS)], "currency": "GBP", "country": "Country 1",
                "region": REGIONS[si % len(REGIONS)],
                "spend": spend, "off": off, "quotes": deals * 2 + rnd.randrange(3), "deals": deals,
                "cycle_days": deals * (18 + rnd.randrange(14)), "reconciled": round(deals * (0.82 + rnd.random() * 0.15)),
                "assessed": assessed, "matched": round(assessed * (0.8 + rnd.random() * 0.18)),
                "saved": round(spend * 0.012 * rnd.random()), "pipeline": round(spend * 0.05 * (0.5 + rnd.random())),
                "inflight": rnd.randrange(3), "dups": 1 if rnd.random() < 0.08 else 0,
                "visible": round(spend * (0.9 if band != "Tail" else 0.55)),
                "compliant": round(spend * (0.78 + 0.04 * rnd.random())),
            })
    return tuple(rows)


def _ratio(n: float, d: float) -> Optional[float]:
    return None if not d else n / d


AGG = {
    "committed_spend": lambda r: float(sum(x["spend"] for x in r)),
    "off_contract_spend": lambda r: float(sum(x["off"] for x in r)),
    "non_po_spend": lambda r: _pct(_ratio(sum(x["off"] for x in r), sum(x["spend"] for x in r))),
    "quote_volume": lambda r: float(sum(x["quotes"] for x in r)),
    "cycle_time_to_po": lambda r: _ratio(sum(x["cycle_days"] for x in r), sum(x["deals"] for x in r)),
    "value_reconciled_rate": lambda r: _pct(_ratio(sum(x["reconciled"] for x in r), sum(x["deals"] for x in r))),
    "three_way_match_rate": lambda r: _pct(_ratio(sum(x["matched"] for x in r), sum(x["assessed"] for x in r))),
    "savings_secured": lambda r: float(sum(x["saved"] for x in r)),
    "opportunity_pipeline": lambda r: float(sum(x["pipeline"] for x in r)),
    "in_flight_negotiations": lambda r: float(sum(x["inflight"] for x in r)),
    "duplicate_risk": lambda r: float(sum(x["dups"] for x in r)),
    "tail_spend_visibility": lambda r: _pct(_ratio(sum(x["visible"] for x in r), sum(x["spend"] for x in r))),
    "compliance_rate": lambda r: _pct(_ratio(sum(x["compliant"] for x in r), sum(x["spend"] for x in r))),
    "tail_spend_breakdown": lambda r: float(sum(x["spend"] for x in r)),
    "spend_by_category": lambda r: float(sum(x["spend"] for x in r)),
}


def _pct(v: Optional[float]) -> Optional[float]:
    return None if v is None else 100.0 * v


def _dim(row: Dict[str, Any], d: str) -> Tuple[Any, Any]:
    m = row["month"]
    if d == "month":
        return m, m.strftime("%b %Y")
    if d == "quarter":
        q = dt.date(m.year, 3 * ((m.month - 1) // 3) + 1, 1)
        return q, f"{m.year} Q{(m.month - 1) // 3 + 1}"
    if d == "year":
        return dt.date(m.year, 1, 1), str(m.year)
    if d == "supplier":
        return row["supplier"], row["supplier_label"]
    return row[d], row[d]


def _risk(supplier: str, dim: str) -> float:
    return round(20 + random.Random(f"risk-{supplier}-{dim}").random() * 70, 1)


def series(metric: R.Metric, group_by: List[str], filters: Dict[str, List[str]],
           start: dt.date, end_inclusive: dt.date, scope: Any = None, today: Optional[dt.date] = None) -> List[Row]:
    today = today or dt.date.today()
    if metric.key == "supplier_risk_profile":
        return _risk_series(group_by, filters)
    fn = AGG.get(metric.key)
    if fn is None:
        raise LookupError(f"{metric.key} has no presentation data")
    # a month row stands for the whole month: include it when the month is inside the window
    rows = [r for r in _facts(today) if _in_window(r["month"], start, end_inclusive)
            and all(str(_dim(r, d)[0]) in set(vs) for d, vs in filters.items())]
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    labels: Dict[Tuple, Tuple] = {}
    for r in rows:
        parts = [_dim(r, d) for d in group_by]
        k = tuple(p[0] for p in parts)
        groups.setdefault(k, []).append(r)
        labels[k] = tuple(p[1] for p in parts)
    if not group_by:
        groups[()] = rows
        labels[()] = ()
    out = []
    for k in sorted(groups, key=lambda x: tuple(str(i) for i in x)):
        v = fn(groups[k]) if groups[k] else None
        out.append(Row(k, labels[k], None if v is None else float(v)))
    return out


def _in_window(month: dt.date, start: dt.date, end: dt.date) -> bool:
    nxt = dt.date(month.year + (month.month == 12), 1 if month.month == 12 else month.month + 1, 1)
    return month <= end and nxt > start


def _risk_series(group_by: List[str], filters: Dict[str, List[str]]) -> List[Row]:
    sups = [_sup(i) for i in range(1, N_SUPPLIERS + 1)]
    sups = [s for s in sups if "supplier" not in filters or s[0] in filters["supplier"]]
    dims = [d for d in RISK_DIMS if "risk_dimension" not in filters or d in filters["risk_dimension"]]
    cells = {(s[0], d): _risk(s[0], d) for s in sups for d in dims}
    names = dict(sups)
    if group_by == ["risk_dimension"]:
        return [Row((d,), (d,), round(sum(cells[(s[0], d)] for s in sups) / len(sups), 1)) for d in dims]
    if group_by == ["supplier"]:
        return [Row((s[0],), (s[1],), round(sum(cells[(s[0], d)] for d in dims) / len(dims), 1)) for s in sups]
    if sorted(group_by) == ["risk_dimension", "supplier"]:
        order = group_by
        out = []
        for s in sups:
            for d in dims:
                kv = {"supplier": (s[0], s[1]), "risk_dimension": (d, d)}
                out.append(Row(tuple(kv[g][0] for g in order), tuple(kv[g][1] for g in order), cells[(s[0], d)]))
        return out
    raise LookupError("supplier_risk_profile is grouped by risk_dimension, supplier, or both")
