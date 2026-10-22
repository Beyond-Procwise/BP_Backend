"""compare_total_cost: landed cost then total cost of ownership for several options, in one call.

Built because the model, handed the two registered calculations separately, put five years of
maintenance into the goods lines of landed cost and added the rest by hand (live, 2026-10-22). This
runs ``landed_cost.total_cost`` and then ``tco.total_cost`` per option through the same
``run_calculation`` path an agent uses (same audit line, same provenance), ranks the options by
present value, and states for each whether its total is complete or a lower bound. It estimates
nothing: a rate it is not given stays missing, and the ranking is said to rest on lower bounds when
any option has a gap.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

log = logging.getLogger(__name__)

MAX_OPTIONS = 6
MIN_OPTIONS = 2
LIFETIME_KEYS = ("key", "amount")


def _run_default(name: str, inputs: Mapping[str, Any]) -> Dict[str, Any]:
    from src.services import calculation_tools

    return calculation_tools.run_calculation(name, inputs)


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate(options: Any, horizon_years: Any, cost_of_capital: Any) -> Optional[str]:
    if not isinstance(options, (list, tuple)) or len(options) < MIN_OPTIONS:
        return f"give at least {MIN_OPTIONS} options, each with a name and its price lines"
    if len(options) > MAX_OPTIONS:
        return f"at most {MAX_OPTIONS} options in one comparison"
    names = []
    for i, o in enumerate(options):
        if not isinstance(o, Mapping):
            return f"option {i + 1} must be an object"
        name = str(o.get("name") or "").strip()
        if not name:
            return f"option {i + 1} has no name"
        if name in names:
            return f"two options are both called {name!r}"
        names.append(name)
        lines = o.get("lines")
        if not isinstance(lines, (list, tuple)) or not lines:
            return f"option {name!r} has no lines (unit_price, quantity, currency)"
        for item in o.get("lifetime_items") or []:
            if not isinstance(item, Mapping) or any(k not in item for k in LIFETIME_KEYS):
                return (f"option {name!r}: every lifetime item needs key and amount, plus year or day "
                        "(maintenance, support, consumables, energy, disposal belong here, never in lines)")
    h = _num(horizon_years)
    if h is None or h <= 0:
        return "horizon_years must be a positive number of years"
    r = _num(cost_of_capital)
    if r is None or not 0.0 <= r <= 1.0:
        return "cost_of_capital must be an annual rate between 0 and 1 (for example 0.12), from policy or finance"
    return None


def compare(options: Any = None, horizon_years: Any = None, cost_of_capital: Any = None,
            buyer_currency: Any = "GBP", fx_rates: Any = None, fx_table_date: Any = None,
            default_insurance_rate: Any = None, *,
            run: Optional[Callable[[str, Mapping[str, Any]], Dict[str, Any]]] = None) -> Dict[str, Any]:
    problem = _validate(options, horizon_years, cost_of_capital)
    if problem:
        return {"found": False, "note": problem + "."}
    runner = run or _run_default
    results: List[Dict[str, Any]] = []
    for o in options:
        name = str(o["name"]).strip()
        landed_inputs: Dict[str, Any] = {
            "lines": o["lines"], "buyer_currency": str(buyer_currency or "GBP").upper(),
            "cross_border": bool(o.get("cross_border", True)),
            "components": list(o.get("components") or []),  # an option with no stated extras has none, not None
        }
        for key in ("incoterm", "duty_rate", "customs_basis", "evaluation_date"):
            if o.get(key) is not None:
                landed_inputs[key] = o[key]
        if fx_rates is not None:
            landed_inputs["fx_rates"] = fx_rates
        if fx_table_date is not None:
            landed_inputs["fx_table_date"] = fx_table_date
        if default_insurance_rate is not None:
            landed_inputs["default_insurance_rate"] = default_insurance_rate
        landed = runner("landed_cost.total_cost", landed_inputs)
        if not landed.get("assessed"):
            results.append({"name": name, "assessed": False, "stage": "landed_cost",
                            "why": landed.get("why") or "the landed cost could not be assessed",
                            "expects": landed.get("expects")})
            continue
        lv = landed.get("value") or {}
        tco_inputs: Dict[str, Any] = {"landed": lv, "cost_of_capital": float(cost_of_capital),
                                      "horizon_years": float(horizon_years)}
        for key in ("payment_terms", "ap_can_take_discount", "lifetime_items"):
            if o.get(key) is not None:
                tco_inputs[key] = o[key]
        tco = runner("tco.total_cost", tco_inputs)
        if not tco.get("assessed"):
            results.append({"name": name, "assessed": False, "stage": "tco", "landed_total": lv.get("landed_total"),
                            "landed_status": lv.get("landed_status"),
                            "why": tco.get("why") or "the ownership cost could not be assessed",
                            "expects": tco.get("expects")})
            continue
        tv = tco.get("value") or {}
        results.append({
            "name": name, "assessed": True,
            "landed_total": lv.get("landed_total"), "landed_status": lv.get("landed_status"),
            "missing": list(lv.get("missing") or []), "not_stated": list(lv.get("not_stated") or []),
            "estimated": list(lv.get("estimated") or []),
            "tco": tv.get("tco"), "tco_status": tv.get("tco_status"),
            "discount_choice": tv.get("discount_choice"),
            "excluded_lifetime": list(tv.get("excluded_lifetime") or []),
            "warnings": list(tv.get("warnings") or []),
            "calculations": [landed.get("calculation"), tco.get("calculation")],
        })

    ranked = sorted((r for r in results if r.get("assessed") and _num(r.get("tco")) is not None),
                    key=lambda r: float(r["tco"]))
    out: Dict[str, Any] = {"found": True, "currency": str(buyer_currency or "GBP").upper(),
                           "horizon_years": float(horizon_years), "cost_of_capital": float(cost_of_capital),
                           "options": results, "ranking": [r["name"] for r in ranked],
                           "cheapest": ranked[0]["name"] if ranked else None, "gaps_pct": {},
                           "lower_bounds": [r["name"] for r in ranked if r.get("tco_status") != "complete"],
                           "unassessed": [r["name"] for r in results if not r.get("assessed")]}
    if len(ranked) >= 2:
        base = float(ranked[0]["tco"])
        for r in ranked[1:]:
            gap = runner("deal.pct_change", {"new": float(r["tco"]), "base": base})
            out["gaps_pct"][r["name"]] = gap.get("value") if gap.get("assessed") else None
    # A component missing for every ranked option alike does not change the order.
    if ranked:
        shared = set(ranked[0]["missing"])
        for r in ranked[1:]:
            shared &= set(r["missing"])
        out["missing_for_all"] = sorted(shared)
    notes = [f"Present value over {float(horizon_years):g} years at the cost of capital given, from "
             "landed_cost.total_cost then tco.total_cost per option; nothing is estimated."]
    if out["lower_bounds"]:
        notes.append("A total marked incomplete is a lower bound: its option lacks a stated figure for the "
                     "components listed under missing. "
                     + ("The components missing for every option alike do not change the ranking. "
                        if out.get("missing_for_all") else "")
                     + "Quote the totals as lower bounds and ask for the missing figures.")
    if out["unassessed"]:
        notes.append("Options under unassessed could not be costed: see each one's why and expects.")
    if len(ranked) < 2:
        notes.append("Fewer than two options could be costed, so there is no ranking.")
    out["note"] = " ".join(notes)
    return out
