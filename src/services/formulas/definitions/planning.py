"""Inventory and forecasting arithmetic from the library sheet (rows 063, 064, 072 to 076), as registered
calculations: economic order quantity, safety stock, reorder point, inventory turns, ABC/XYZ classes,
exponential smoothing and forecast accuracy.

Each takes the figures a person or a tool supplies (the platform holds no stock, no demand series and no
BOM of its own) and returns None rather than a number when the inputs cannot support one. The service
level for safety stock is a stated percentage mapped through a fixed table of normal-distribution factors;
anything off the table is refused rather than interpolated.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Optional, Sequence

from ..contract import COUNT, MONEY, PERCENT, RATIO, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "analytics"
_FROM = date(2026, 10, 22)

#: Service level (percent) to the one-sided normal factor z, the usual planning table.
SERVICE_FACTORS = {80.0: 0.842, 85.0: 1.036, 90.0: 1.282, 95.0: 1.645, 97.5: 1.960, 98.0: 2.054, 99.0: 2.326, 99.5: 2.576, 99.9: 3.090}


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out or out in (float("inf"), float("-inf")) else out


@formula(
    "inventory.eoq",
    version="1.0.0",
    owner=_OWNER,
    purpose="Economic order quantity: the order size that balances ordering cost against holding cost",
    effective_from=_FROM,
    inputs=[
        Term("annual_demand", COUNT, "units per year", required=False, minimum=0.0),
        Term("order_cost", MONEY, "cost of placing one order, from policy or finance", required=False, minimum=0.0),
        Term("holding_cost_per_unit", MONEY, "cost of holding one unit for a year, from policy or finance", required=False, minimum=0.0),
    ],
    output=Output("float", COUNT, "sqrt(2 * annual_demand * order_cost / holding_cost_per_unit), 1dp; None when holding cost is zero or an input is absent"),
    notes="The classic Wilson formula; it assumes steady demand and no price breaks. A price break is compared separately with deal.scaled_total and finance.carrying_cost.",
    golden=[
        GoldenVector(inputs={"annual_demand": 12000, "order_cost": 50, "holding_cost_per_unit": 2.5}, expected=692.8, tolerance=0.05),
        GoldenVector(inputs={"annual_demand": 0, "order_cost": 50, "holding_cost_per_unit": 2.5}, expected=0.0),
        GoldenVector(inputs={"annual_demand": 12000, "order_cost": 50, "holding_cost_per_unit": 0}, expected=None),
        GoldenVector(inputs={"annual_demand": None, "order_cost": 50, "holding_cost_per_unit": 2.5}, expected=None),
    ],
)
def eoq(annual_demand=None, order_cost=None, holding_cost_per_unit=None):
    d, s, h = _num(annual_demand), _num(order_cost), _num(holding_cost_per_unit)
    if d is None or s is None or h is None or h <= 0 or d < 0 or s < 0:
        return None
    return round(math.sqrt(2.0 * d * s / h), 1)


@formula(
    "inventory.safety_stock",
    version="1.0.0",
    owner=_OWNER,
    purpose="Safety stock for a stated service level from demand variability over the lead time",
    effective_from=_FROM,
    inputs=[
        Term("demand_sd_per_period", COUNT, "standard deviation of demand per period, as the user or a series calculation states it", required=False, minimum=0.0),
        Term("lead_time_periods", COUNT, "lead time in the same periods", required=False, minimum=0.0),
        Term("service_level_pct", PERCENT, "target service level; one of 80, 85, 90, 95, 97.5, 98, 99, 99.5, 99.9", required=False),
    ],
    output=Output("float", COUNT, "z * demand_sd * sqrt(lead_time), 1dp; None when the service level is off the table or an input is absent"),
    notes="A service level off the fixed table is refused, not interpolated; the table is the usual one-sided normal factors.",
    golden=[
        GoldenVector(inputs={"demand_sd_per_period": 40, "lead_time_periods": 4, "service_level_pct": 95}, expected=131.6, tolerance=0.05),
        GoldenVector(inputs={"demand_sd_per_period": 0, "lead_time_periods": 4, "service_level_pct": 95}, expected=0.0),
        GoldenVector(inputs={"demand_sd_per_period": 40, "lead_time_periods": 4, "service_level_pct": 93}, expected=None),
        GoldenVector(inputs={"demand_sd_per_period": 40, "lead_time_periods": None, "service_level_pct": 95}, expected=None),
    ],
)
def safety_stock(demand_sd_per_period=None, lead_time_periods=None, service_level_pct=None):
    sd, lt, sl = _num(demand_sd_per_period), _num(lead_time_periods), _num(service_level_pct)
    if sd is None or lt is None or sl is None or sd < 0 or lt < 0:
        return None
    z = SERVICE_FACTORS.get(round(sl, 1))
    if z is None:
        return None
    return round(z * sd * math.sqrt(lt), 1)


@formula(
    "inventory.reorder_point",
    version="1.0.0",
    owner=_OWNER,
    purpose="Reorder point: demand over the lead time plus safety stock",
    effective_from=_FROM,
    inputs=[
        Term("demand_per_period", COUNT, "average demand per period", required=False, minimum=0.0),
        Term("lead_time_periods", COUNT, "lead time in the same periods", required=False, minimum=0.0),
        Term("safety_stock", COUNT, "safety stock in units (inventory.safety_stock or the user's figure)", required=False, minimum=0.0),
    ],
    output=Output("float", COUNT, "demand_per_period * lead_time_periods + safety_stock, 1dp; None when demand or lead time is absent"),
    notes="Safety stock absent is taken as zero and said so by the caller; demand or lead time absent is None.",
    golden=[
        GoldenVector(inputs={"demand_per_period": 100, "lead_time_periods": 3, "safety_stock": 131.6}, expected=431.6),
        GoldenVector(inputs={"demand_per_period": 100, "lead_time_periods": 3}, expected=300.0),
        GoldenVector(inputs={"demand_per_period": None, "lead_time_periods": 3, "safety_stock": 10}, expected=None),
    ],
)
def reorder_point(demand_per_period=None, lead_time_periods=None, safety_stock=None):
    d, lt = _num(demand_per_period), _num(lead_time_periods)
    ss = _num(safety_stock) or 0.0
    if d is None or lt is None or d < 0 or lt < 0 or ss < 0:
        return None
    return round(d * lt + ss, 1)


@formula(
    "inventory.turns",
    version="1.0.0",
    owner=_OWNER,
    purpose="Inventory turns and days of inventory from cost of goods sold and average inventory value",
    effective_from=_FROM,
    inputs=[
        Term("cost_of_goods_sold", MONEY, "cost of goods sold over the period", required=False, minimum=0.0),
        Term("average_inventory_value", MONEY, "average inventory value over the same period", required=False, minimum=0.0),
        Term("period_days", COUNT, "days in the period (365 for a year)", required=False, minimum=0.0),
    ],
    output=Output("dict", RATIO, "{turns, days_of_inventory}; None when inventory or the period is zero or an input is absent"),
    notes="Turns = COGS / average inventory; days = period_days / turns. Both figures are the user's or finance's.",
    golden=[
        GoldenVector(inputs={"cost_of_goods_sold": 1200000, "average_inventory_value": 200000, "period_days": 365}, expected={"turns": 6.0, "days_of_inventory": 60.8}),
        GoldenVector(inputs={"cost_of_goods_sold": 1200000, "average_inventory_value": 0, "period_days": 365}, expected=None),
        GoldenVector(inputs={"cost_of_goods_sold": 0, "average_inventory_value": 200000, "period_days": 365}, expected=None),
    ],
)
def turns(cost_of_goods_sold=None, average_inventory_value=None, period_days=None):
    c, inv, days = _num(cost_of_goods_sold), _num(average_inventory_value), _num(period_days)
    if c is None or inv is None or days is None or c <= 0 or inv <= 0 or days <= 0:
        return None
    t = c / inv
    return {"turns": round(t, 2), "days_of_inventory": round(days / t, 1)}


@formula(
    "inventory.abc_xyz",
    version="1.0.0",
    owner=_OWNER,
    purpose="ABC class by share of annual value (cumulative 80/95 percent) and XYZ class by demand variability (coefficient of variation)",
    effective_from=_FROM,
    inputs=[
        Term("items", ROWS, "rows of {item, annual_value, demand: [per-period quantities]}"),
        Term("a_cutoff_pct", PERCENT, "cumulative value share for class A", required=False, minimum=0.0),
        Term("b_cutoff_pct", PERCENT, "cumulative value share for class B", required=False, minimum=0.0),
        Term("x_cv_max", RATIO, "coefficient of variation at or below which demand is X", required=False, minimum=0.0),
        Term("y_cv_max", RATIO, "coefficient of variation at or below which demand is Y", required=False, minimum=0.0),
    ],
    output=Output("list", COUNT, "one row per item: item, annual_value, value_share_pct, abc, cv, xyz; None when no item has a positive value"),
    notes=(
        "Defaults when the cutoffs are absent: A up to 80 percent of cumulative value, B up to 95, X at CV up to 0.5, Y up to 1.0. "
        "An item with fewer than two demand periods has no CV and is class Z (unknown variability), said so by the caller."
    ),
    golden=[
        GoldenVector(
            inputs={"items": [{"item": "P1", "annual_value": 700, "demand": [10, 10, 10, 10]},
                              {"item": "P2", "annual_value": 200, "demand": [5, 15, 5, 15]},
                              {"item": "P3", "annual_value": 100, "demand": [0, 20]}]},
            expected=[{"item": "P1", "annual_value": 700.0, "value_share_pct": 70.0, "abc": "A", "cv": 0.0, "xyz": "X"},
                      {"item": "P2", "annual_value": 200.0, "value_share_pct": 20.0, "abc": "B", "cv": 0.577, "xyz": "Y"},
                      {"item": "P3", "annual_value": 100.0, "value_share_pct": 10.0, "abc": "C", "cv": 1.414, "xyz": "Z"}],
        ),
        GoldenVector(inputs={"items": [{"item": "P1", "annual_value": 0, "demand": [1]}]}, expected=None),
    ],
)
def abc_xyz(items: Sequence[Mapping[str, Any]] = (), a_cutoff_pct=None, b_cutoff_pct=None, x_cv_max=None, y_cv_max=None):
    a_cut = _num(a_cutoff_pct) if a_cutoff_pct is not None else 80.0
    b_cut = _num(b_cutoff_pct) if b_cutoff_pct is not None else 95.0
    x_max = _num(x_cv_max) if x_cv_max is not None else 0.5
    y_max = _num(y_cv_max) if y_cv_max is not None else 1.0
    rows = []
    for row in items or ():
        v = _num((row or {}).get("annual_value"))
        if v is None or v < 0:
            continue
        rows.append((str((row or {}).get("item")), v, list((row or {}).get("demand") or [])))
    total = sum(v for _, v, _ in rows)
    if not rows or total <= 0:
        return None
    rows.sort(key=lambda r: -r[1])
    out, cumulative = [], 0.0
    for item, v, demand in rows:
        share = v / total * 100.0
        cumulative += share
        abc = "A" if cumulative <= a_cut + 1e-9 else ("B" if cumulative <= b_cut + 1e-9 else "C")
        series = [x for x in (_num(d) for d in demand) if x is not None]
        if len(series) >= 2 and sum(series) > 0:
            mean = sum(series) / len(series)
            sd = math.sqrt(sum((x - mean) ** 2 for x in series) / (len(series) - 1))
            cv = sd / mean
            xyz = "X" if cv <= x_max else ("Y" if cv <= y_max else "Z")
        else:
            cv, xyz = None, "Z"
        out.append({"item": item, "annual_value": round(v, 2), "value_share_pct": round(share, 1), "abc": abc,
                    "cv": None if cv is None else round(cv, 3), "xyz": xyz})
    return out


@formula(
    "forecast.exponential_smoothing",
    version="1.0.0",
    owner=_OWNER,
    purpose="Simple exponential smoothing of a demand series, with the next-period forecast",
    effective_from=_FROM,
    inputs=[
        Term("series", ROWS, "the observed values in time order, oldest first"),
        Term("alpha", RATIO, "smoothing factor between 0 and 1, from the planner", required=False, minimum=0.0),
    ],
    output=Output("dict", COUNT, "{forecast, level, periods}; the forecast for the next period, 2dp; None when the series is empty or alpha is off range"),
    notes="No trend or seasonality: it is the flat baseline a planner starts from, and the alpha is the planner's, never chosen here. Default alpha 0.3 when absent, said so by the caller.",
    golden=[
        GoldenVector(inputs={"series": [100, 120, 110, 130], "alpha": 0.5}, expected={"forecast": 120.0, "level": 120.0, "periods": 4}),
        GoldenVector(inputs={"series": [50], "alpha": 0.3}, expected={"forecast": 50.0, "level": 50.0, "periods": 1}),
        GoldenVector(inputs={"series": [], "alpha": 0.3}, expected=None),
        GoldenVector(inputs={"series": [1, 2], "alpha": 1.5}, expected=None),
    ],
)
def exponential_smoothing(series: Sequence[Any] = (), alpha=None):
    values = [x for x in (_num(v) for v in (series or ())) if x is not None]
    a = _num(alpha) if alpha is not None else 0.3
    if not values or a is None or not 0.0 < a <= 1.0:
        return None
    level = values[0]
    for v in values[1:]:
        level = a * v + (1.0 - a) * level
    return {"forecast": round(level, 2), "level": round(level, 2), "periods": len(values)}


@formula(
    "forecast.accuracy",
    version="1.0.0",
    owner=_OWNER,
    purpose="Forecast accuracy: mean absolute percentage error, weighted MAPE and bias over paired actuals and forecasts",
    effective_from=_FROM,
    inputs=[
        Term("actuals", ROWS, "actual values in time order"),
        Term("forecasts", ROWS, "forecast values for the same periods, same order"),
    ],
    output=Output("dict", PERCENT, "{mape_pct, wmape_pct, bias_pct, periods}; None when the lists differ in length, are empty, or actuals sum to zero"),
    notes="MAPE skips periods with a zero actual (and says how many); WMAPE and bias use the sums, so a zero actual is fine there. Bias positive means the forecast ran high.",
    golden=[
        GoldenVector(inputs={"actuals": [100, 200, 300], "forecasts": [110, 180, 300]}, expected={"mape_pct": 6.67, "wmape_pct": 5.0, "bias_pct": -1.67, "periods": 3, "zero_actuals_skipped": 0}, tolerance=0.01),
        GoldenVector(inputs={"actuals": [0, 100], "forecasts": [10, 100]}, expected={"mape_pct": 0.0, "wmape_pct": 10.0, "bias_pct": 10.0, "periods": 2, "zero_actuals_skipped": 1}),
        GoldenVector(inputs={"actuals": [1, 2], "forecasts": [1]}, expected=None),
        GoldenVector(inputs={"actuals": [], "forecasts": []}, expected=None),
    ],
)
def accuracy(actuals: Sequence[Any] = (), forecasts: Sequence[Any] = ()):
    a = [_num(v) for v in (actuals or ())]
    f = [_num(v) for v in (forecasts or ())]
    if not a or len(a) != len(f) or any(v is None for v in a) or any(v is None for v in f):
        return None
    total = sum(a)
    if total <= 0:
        return None
    abs_err = sum(abs(x - y) for x, y in zip(a, f))
    pct = [abs(x - y) / abs(x) for x, y in zip(a, f) if x != 0]
    return {"mape_pct": round(sum(pct) / len(pct) * 100.0, 2) if pct else 0.0,
            "wmape_pct": round(abs_err / total * 100.0, 2),
            "bias_pct": round((sum(f) - total) / total * 100.0, 2),
            "periods": len(a), "zero_actuals_skipped": len(a) - len(pct)}
