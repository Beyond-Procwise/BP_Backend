"""Total cost of ownership: the present value of every cash flow to acquire, run and dispose of
the goods over the horizon, minus inflows. Built on a landed-cost result.

Pure arithmetic, no I/O. ``PV(amount, days) = amount / (1 + r) ** (days / 365)``, with days
counted from the evaluation date (delivery). A timing nobody stated is day 0: no financing
benefit is claimed for a payment date that was never given.

Landed cost ignores timing; this is where timing is applied. Capital cost is ``r`` alone: no
lifetime item may carry its own financing charge (inventory carrying cost excludes capital).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .landed_cost import COUNTED, MISSING

#: The ownership checklist. Every item is listed in the result with a status, so a gap is
#: visible, but only an item that was supplied enters the total.
OWNERSHIP_CHECKLIST: Sequence[tuple] = (
    ("acquisition", ("sourcing_effort", "supplier_onboarding", "po_invoice_processing")),
    ("deployment", ("site_preparation", "installation", "commissioning", "integration",
                    "initial_training")),
    ("operation", ("licences_subscriptions", "consumables", "energy_utilities",
                   "maintenance_support", "spares", "warranty_extension", "ongoing_training")),
    ("quality", ("receiving_inspection", "rework", "returns", "scrap")),
    ("inventory", ("storage", "handling", "obsolescence", "safety_stock")),
    ("contract", ("price_escalation", "volume_rebates")),
    ("fx", ("hedging_cost",)),
    ("end_of_life", ("decommissioning", "data_destruction", "disposal", "residual_value")),
    ("exit", ("switching_cost",)),
)
_KNOWN = {k for _g, keys in OWNERSHIP_CHECKLIST for k in keys}
_YEAR_DAYS = 365.0


def present_value(amount: float, days: float, rate: float) -> float:
    return amount / (1.0 + rate) ** (days / _YEAR_DAYS)


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def tco(
    *, landed: Mapping[str, Any], cost_of_capital: float, horizon_years: float,
    payment_terms: Optional[Mapping[str, Any]] = None,
    ap_can_take_discount: Optional[bool] = None,
    lifetime_items: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """TCO from a landed-cost result. Lifetime items must already be in the landed currency."""
    r = float(cost_of_capital)
    horizon_days = float(horizon_years) * _YEAR_DAYS
    comps: List[Dict[str, Any]] = []
    warnings: List[str] = []

    terms = dict(payment_terms or {})
    discount_choice = "net"
    for c in landed["components"]:
        if c["status"] not in COUNTED:
            comps.append({**c, "pv": None})
            continue
        amount = c["amount"]
        if c["name"] == "goods":
            net_days = _num(terms.get("net_days")) or 0.0
            pv = present_value(amount, net_days, r)
            d, dd = _num(terms.get("discount_pct")), _num(terms.get("discount_days"))
            if d is not None and dd is not None:
                pv_disc = present_value(amount * (1.0 - d), dd, r)
                if ap_can_take_discount is None:
                    warnings.append("ap_can_take_discount is not stated: the discount is not assumed")
                if ap_can_take_discount and pv_disc < pv:
                    pv, discount_choice = pv_disc, "discount"
            comps.append({**c, "pay_day": net_days if discount_choice == "net" else dd, "pv": pv})
            continue
        day = c.get("pay_day")
        day = 0.0 if day is None else float(day)
        comps.append({**c, "pay_day": day, "pv": present_value(amount, day, r)})

    outflows = sum(c["pv"] for c in comps if c["pv"] is not None)

    inflows = 0.0
    supplied: set = set()
    excluded: List[str] = []
    beyond: List[str] = []
    life_rows: List[Dict[str, Any]] = []
    for item in lifetime_items:
        key = item.get("key")
        amount = _num(item.get("amount"))
        if key not in _KNOWN:
            excluded.append(f"{key} (not on the ownership checklist)")
            continue
        if item.get("currency") and str(item["currency"]).upper() != landed["currency"]:
            raise ValueError(f"lifetime item {key!r} is in {item['currency']}: convert it to "
                             f"{landed['currency']} first")
        day = _num(item.get("day"))
        if day is None and _num(item.get("year")) is not None:
            day = _num(item.get("year")) * _YEAR_DAYS
        if amount is None or day is None:
            continue  # no amount or no timing: it stays on the unstated list
        supplied.add(key)
        if day > horizon_days:
            beyond.append(key)
            continue
        inflow = str(item.get("direction") or "outflow") == "inflow"
        pv = present_value(amount, day, r)
        inflows += pv if inflow else 0.0
        outflows += 0.0 if inflow else pv
        life_rows.append({"group": item.get("group"), "name": key, "status": "stated",
                          "amount": amount, "direction": "inflow" if inflow else "outflow",
                          "pay_day": day, "pv": pv})
    unstated = [k for _g, keys in OWNERSHIP_CHECKLIST for k in keys if k not in supplied]

    return {
        **{k: landed[k] for k in ("currency", "fx_table_date", "incoterm")},
        "r": r, "horizon_years": float(horizon_years),
        "components": comps, "lifetime_items": life_rows,
        "landed_total": landed["landed_total"], "unit_landed": landed["unit_landed"],
        "landed_status": landed["landed_status"], "missing": landed["missing"],
        "estimated": landed["estimated"],
        "discount_choice": discount_choice,
        "tco": outflows - inflows,
        "tco_status": landed["landed_status"],
        "excluded_lifetime": unstated,
        "beyond_horizon": beyond,
        "unrecognised_lifetime_items": [e for e in excluded],
        "warnings": warnings,
    }
