"""Should-cost of a manufactured part (SC-012): a bottom-up, modelled estimate.

Material, conversion, labour, overhead, SG&A, logistics and margin per unit, every rate listed
with where it came from. It is a modelled estimate and is labelled so: it is never a supplier's
actual cost. A rate nobody supplied is an error, not a default.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

METHOD_VERSION = "1.0.0"
DRIVERS = ("material_price", "scrap_rate", "cycle_time", "machine_rate", "labour_rate",
           "overhead_pct", "sga_pct", "margin_pct", "logistics")
_MACHINE_KEYS = ("capex", "life_years", "interest_rate", "floor_m2", "cost_per_m2", "kw", "load_factor",
                 "energy_price", "available_hours", "maintenance_pct")
_FLAG_OVER = 0.10


class ShouldCostError(ValueError):
    """The inputs cannot be costed; the message says what to fix."""


def _f(value: Any, what: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ShouldCostError(f"{what} is missing or not a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ShouldCostError(f"{what} is not a number") from None
    if out != out or abs(out) == float("inf"):
        raise ShouldCostError(f"{what} is not a finite number")
    return out


def _machine_rate(op: Mapping[str, Any], i: int, assumptions: List[str]):
    if op.get("machine_rate") is not None:
        return _f(op["machine_rate"], f"machine_rate of operation {i + 1}"), "supplied"
    m = op.get("machine")
    if not isinstance(m, Mapping):
        raise ShouldCostError(f"operation {i + 1} needs machine_rate or the machine inputs to derive one")
    v = {k: _f(m.get(k), f"machine.{k} of operation {i + 1}") for k in _MACHINE_KEYS}
    if v["available_hours"] <= 0 or v["life_years"] <= 0:
        raise ShouldCostError(f"operation {i + 1}: available_hours and life_years must be above zero")
    rate = (v["capex"] / v["life_years"] + v["capex"] * v["interest_rate"] / 2 + v["floor_m2"] * v["cost_per_m2"]
            + v["kw"] * v["load_factor"] * v["energy_price"] * v["available_hours"]
            + v["capex"] * v["maintenance_pct"]) / v["available_hours"]
    assumptions.append(f"The machine rate of operation {i + 1} was derived from the machine inputs, not supplied.")
    return rate, "derived from machine inputs"


def _validate(bom, routing, batch_size):
    if not bom:
        raise ShouldCostError("the bill of materials is empty")
    if not routing:
        raise ShouldCostError("the routing is empty")
    if _f(batch_size, "batch_size") <= 0:
        raise ShouldCostError("batch_size must be above zero")
    for j, line in enumerate(bom):
        if line.get("bought_in_price") is not None:
            _f(line["bought_in_price"], f"bought_in_price of line {j + 1}")
            continue
        if _f(line.get("net_weight"), f"net_weight of line {j + 1}") <= 0:
            raise ShouldCostError(f"net_weight of line {j + 1} must be above zero")
        _f(line.get("price_per_kg"), f"price_per_kg of line {j + 1}")
        scrap = _f(line.get("scrap_rate", 0.0), f"scrap_rate of line {j + 1}")
        if not 0.0 <= scrap < 1.0:
            raise ShouldCostError(f"scrap_rate of line {j + 1} must be from 0 up to but not including 1")
    for i, op in enumerate(routing):
        if _f(op.get("cycle_time_s"), f"cycle_time_s of operation {i + 1}") <= 0:
            raise ShouldCostError(f"operation {i + 1}: the cycle time must be above zero")
        if _f(op.get("cavities"), f"cavities of operation {i + 1}") < 1:
            raise ShouldCostError(f"operation {i + 1}: cavities must be at least 1")
        oee = _f(op.get("oee"), f"oee of operation {i + 1}")
        if not 0.0 < oee <= 1.0:
            raise ShouldCostError(f"operation {i + 1}: oee must be above 0 and at most 1")
        _f(op.get("setup_time_h", 0.0), f"setup_time_h of operation {i + 1}")
        ops = _f(op.get("operators_per_machine"), f"operators_per_machine of operation {i + 1}")
        if ops > 0 and op.get("labour_rate") is None:
            raise ShouldCostError(f"labour_rate is required for operation {i + 1}; no country rate is assumed")


def _cost(bom, routing, batch, oh, sga, margin, logistics, margin_base, sc, rates, assumptions):
    material = 0.0
    for line in bom:
        if line.get("bought_in_price") is not None:
            material += float(line["bought_in_price"]) * sc["material_price"]
            continue
        net = float(line["net_weight"])
        price = float(line["price_per_kg"]) * sc["material_price"]
        scrap = min(0.999999, float(line.get("scrap_rate", 0.0)) * sc["scrap_rate"])
        recovery = float(line.get("scrap_recovery_price_pct", 0.0))
        gross = net / (1.0 - scrap)
        material += gross * price - (gross - net) * price * recovery
    conversion = labour = 0.0
    ops_out = []
    for i, op in enumerate(routing):
        rate, _ = rates[i]
        rate *= sc["machine_rate"]
        hours = (float(op["cycle_time_s"]) * sc["cycle_time"] / 3600.0) / (float(op["cavities"]) * float(op["oee"]))
        conv = hours * rate + float(op.get("setup_time_h", 0.0)) * rate / batch
        lab = hours * float(op["operators_per_machine"]) * float(op.get("labour_rate") or 0.0) * sc["labour_rate"]
        conversion += conv
        labour += lab
        ops_out.append({"process": op.get("process"), "machine_rate": rate, "hours_per_unit": hours,
                        "conversion": conv, "labour": lab})
    overhead = oh * sc["overhead_pct"] * (conversion + labour)
    manufacturing = material + conversion + labour + overhead
    sga_cost = sga * sc["sga_pct"] * manufacturing
    log_cost = logistics * sc["logistics"]
    base = (material + conversion + labour + overhead + sga_cost + log_cost) if margin_base == "total_cost" \
        else (conversion + labour + overhead + sga_cost)
    margin_cost = margin * sc["margin_pct"] * base
    elements = {"material": material, "conversion": conversion, "labour": labour, "overhead": overhead,
                "sga": sga_cost, "logistics": log_cost, "margin": margin_cost}
    return elements, sum(elements.values()), ops_out


def _pct(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def should_cost(
    bom: Sequence[Mapping[str, Any]], routing: Sequence[Mapping[str, Any]], batch_size: float,
    overhead_pct: float, sga_pct: float, margin_pct: float, logistics_per_unit: float = 0.0,
    currency: str = "GBP", margin_base: str = "value_add_only", annual_volume: Optional[float] = None,
    sensitivity_pct: float = 10.0, monte_carlo_n: int = 0, seed: Optional[int] = None,
    distributions: Optional[Mapping[str, Mapping[str, float]]] = None,
    quote_breakdown: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    if margin_base not in ("value_add_only", "total_cost"):
        raise ShouldCostError("margin_base must be value_add_only or total_cost")
    if monte_carlo_n and seed is None:
        raise ShouldCostError("monte carlo needs a seed so the result can be reproduced")
    _validate(bom, routing, batch_size)
    oh, sga, margin = _f(overhead_pct, "overhead_pct"), _f(sga_pct, "sga_pct"), _f(margin_pct, "margin_pct")
    logistics = _f(logistics_per_unit, "logistics_per_unit")
    batch = float(batch_size)
    assumptions: List[str] = ["Prices and rates are those supplied by the caller; no market index or country rate table was used."]
    for j, line in enumerate(bom):
        if line.get("bought_in_price") is None and float(line.get("scrap_rate", 0.0)) > 0 \
                and line.get("scrap_recovery_price_pct") is None:
            assumptions.append(f"Scrap on bill of materials line {j + 1} is valued at nothing: no recovery share was given.")
    rates = [_machine_rate(op, i, assumptions) for i, op in enumerate(routing)]
    one = {d: 1.0 for d in DRIVERS}
    elements, total, ops_out = _cost(bom, routing, batch, oh, sga, margin, logistics, margin_base, one, rates, assumptions)
    rates_used = [{"name": "machine_rate", "operation": i + 1, "value": r, "source": src} for i, (r, src) in enumerate(rates)]
    rates_used += [{"name": "labour_rate", "operation": i + 1, "value": float(op["labour_rate"]), "source": "supplied"}
                   for i, op in enumerate(routing) if op.get("labour_rate") is not None]
    out: Dict[str, Any] = {
        "label": "modelled estimate", "currency": currency, "unit_cost": total, "per_unit": elements,
        "per_unit_share": {k: (v / total if total else None) for k, v in elements.items()},
        "operations": ops_out, "rates_used": rates_used, "assumptions": assumptions, "margin_base": margin_base,
        "method_version": METHOD_VERSION,
    }
    if annual_volume is not None:
        vol = _f(annual_volume, "annual_volume")
        out["annual"] = {"volume": vol, "total": total * vol, "by_element": {k: v * vol for k, v in elements.items()}}

    s = _f(sensitivity_pct, "sensitivity_pct") / 100.0
    tornado = []
    for d in DRIVERS:
        lo = _cost(bom, routing, batch, oh, sga, margin, logistics, margin_base, {**one, d: 1 - s}, rates, [])[1]
        hi = _cost(bom, routing, batch, oh, sga, margin, logistics, margin_base, {**one, d: 1 + s}, rates, [])[1]
        tornado.append({"driver": d, "low": lo, "high": hi, "swing": abs(hi - lo),
                        "elasticity": ((hi - lo) / total) / (2 * s) if total and s else None})
    tornado.sort(key=lambda t: (-t["swing"], t["driver"]))
    mc = None
    if monte_carlo_n:
        rng = random.Random(int(seed))
        dist = dict(distributions or {})
        for d, spec in dist.items():
            if d not in DRIVERS:
                raise ShouldCostError(f"distribution for {d}: drivers are {', '.join(DRIVERS)}")
            if not (spec["min"] <= spec["mode"] <= spec["max"]):
                raise ShouldCostError(f"distribution for {d}: need min <= mode <= max")
        draws = []
        for _ in range(int(monte_carlo_n)):
            sc = dict(one)
            for d, spec in dist.items():
                sc[d] = rng.triangular(spec["min"], spec["max"], spec["mode"]) if spec["max"] > spec["min"] else spec["min"]
            draws.append(_cost(bom, routing, batch, oh, sga, margin, logistics, margin_base, sc, rates, [])[1])
        mc = {"n": int(monte_carlo_n), "seed": int(seed), "p10": _pct(draws, 0.10), "p50": _pct(draws, 0.50),
              "p90": _pct(draws, 0.90)}
    out["sensitivity"] = {"sensitivity_pct": s * 100, "tornado": tornado, "monte_carlo": mc}

    if quote_breakdown:
        gaps: Dict[str, Any] = {}
        for k, v in elements.items():
            if k in quote_breakdown:
                q = float(quote_breakdown[k])
                gaps[k] = {"quote": q, "should_cost": v, "gap": q - v, "flag": bool(v > 0 and q > v * (1 + _FLAG_OVER))}
        tot = None
        if "total" in quote_breakdown:
            q = float(quote_breakdown["total"])
            tot = {"quote": q, "should_cost": total, "gap": q - total, "flag": bool(q > total * (1 + _FLAG_OVER))}
        out["quote_gap"] = {"elements": gaps, "total": tot, "flag_over_pct": _FLAG_OVER * 100}
    return out
