"""Plain finance arithmetic the skills kept doing in their heads: an average per period, a simple payback,
a net present value of dated cash flows.

Registered on 2026-10-22 after the skills content review: business-case asked the registry for a payback
it did not have, budget tracking for a run rate, ordering for a consumption rate. Each is a calculation
the verifier can trace; each returns None rather than a number when the inputs cannot support one.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

from ..contract import COUNT, MONEY, PERCENT, RATIO, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "analytics"
_FROM = date(2026, 10, 22)


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out or out in (float("inf"), float("-inf")) else out


@formula(
    "finance.average_per_period",
    version="1.0.0",
    owner=_OWNER,
    purpose="A total spread evenly over a number of periods (a monthly run rate from a year-to-date figure, a weekly consumption from a quarter's quantity)",
    effective_from=_FROM,
    inputs=[
        Term("period_total", MONEY, "the total over the periods, in money or units", required=False),
        Term("periods", COUNT, "how many periods the total covers", required=False, minimum=0.0),
    ],
    output=Output("float", MONEY, "period_total / periods, 2dp; None when periods is zero or absent"),
    notes="An average is not a forecast: it says nothing about seasonality. The caller names which periods it covers.",
    golden=[
        GoldenVector(inputs={"period_total": 1200.0, "periods": 12}, expected=100.0),
        GoldenVector(inputs={"period_total": 2232.0, "periods": 7}, expected=318.86, tolerance=0.01),
        GoldenVector(inputs={"period_total": 100.0, "periods": 0}, expected=None),
        GoldenVector(inputs={"period_total": None, "periods": 3}, expected=None),
    ],
)
def average_per_period(period_total=None, periods=None):
    t, n = _num(period_total), _num(periods)
    if t is None or n is None or n <= 0:
        return None
    return round(t / n, 2)


@formula(
    "finance.payback_years",
    version="1.0.0",
    owner=_OWNER,
    purpose="Simple payback: years for an annual benefit to repay an initial cost, undiscounted",
    effective_from=_FROM,
    inputs=[
        Term("initial_cost", MONEY, "the one-off cost to recover", required=False, minimum=0.0),
        Term("annual_benefit", MONEY, "the benefit per year, net of running costs, as its owner states it", required=False),
    ],
    output=Output("float", COUNT, "years, 2dp; None when the benefit is zero or negative or an input is absent"),
    notes=(
        "Undiscounted by design: it is the figure finance templates ask for beside the present value, not a "
        "substitute for it. A benefit of zero or less never pays back, and that is None, not infinity."
    ),
    golden=[
        GoldenVector(inputs={"initial_cost": 120000.0, "annual_benefit": 40000.0}, expected=3.0),
        GoldenVector(inputs={"initial_cost": 100000.0, "annual_benefit": 30000.0}, expected=3.33, tolerance=0.01),
        GoldenVector(inputs={"initial_cost": 100000.0, "annual_benefit": 0.0}, expected=None),
        GoldenVector(inputs={"initial_cost": 100000.0, "annual_benefit": -5.0}, expected=None),
        GoldenVector(inputs={"initial_cost": None, "annual_benefit": 10.0}, expected=None),
    ],
)
def payback_years(initial_cost=None, annual_benefit=None):
    c, b = _num(initial_cost), _num(annual_benefit)
    if c is None or b is None or b <= 0 or c < 0:
        return None
    return round(c / b, 2)


@formula(
    "finance.carrying_cost",
    version="1.0.0",
    owner=_OWNER,
    purpose="What holding or releasing an amount for a number of days is worth at an annual rate (a payment-terms change, an early-payment discount, stock held)",
    effective_from=_FROM,
    inputs=[
        Term("amount", MONEY, "the amount carried", required=False, minimum=0.0),
        Term("annual_rate", RATIO, "the annual rate as a fraction, from the governed policy or finance", required=False, minimum=0.0),
        Term("days", COUNT, "days the amount is carried", required=False, minimum=0.0),
    ],
    output=Output("float", MONEY, "amount * annual_rate * days / 365, 2dp; None when an input is absent"),
    notes=(
        "Simple, not compounded: the figure finance quotes for a terms change. Registered because the payment-terms "
        "skill had been feeding a spend figure into tco.total_cost, which needs a landed-cost result and so never ran."
    ),
    golden=[
        GoldenVector(inputs={"amount": 1000000.0, "annual_rate": 0.08, "days": 30}, expected=6575.34, tolerance=0.01),
        GoldenVector(inputs={"amount": 50000.0, "annual_rate": 0.1, "days": 365}, expected=5000.0),
        GoldenVector(inputs={"amount": 50000.0, "annual_rate": 0.1, "days": 0}, expected=0.0),
        GoldenVector(inputs={"amount": None, "annual_rate": 0.1, "days": 30}, expected=None),
    ],
)
def carrying_cost(amount=None, annual_rate=None, days=None):
    a, r, d = _num(amount), _num(annual_rate), _num(days)
    if a is None or r is None or d is None or a < 0 or r < 0 or d < 0:
        return None
    return round(a * r * d / 365.0, 2)


@formula(
    "finance.early_payment_decision",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether an early-payment discount beats paying at net days: the discount's annualised return against the cost of capital",
    effective_from=_FROM,
    inputs=[
        Term("discount_pct", PERCENT, "the discount offered, in percent", required=False, minimum=0.0),
        Term("discount_days", COUNT, "days within which the discount applies", required=False, minimum=0.0),
        Term("net_days", COUNT, "days to pay without the discount", required=False, minimum=0.0),
        Term("annual_rate", RATIO, "the cost of capital as a fraction, from the governed policy or finance", required=False, minimum=0.0),
    ],
    output=Output("dict", PERCENT, "{annualised_return_pct, cost_of_capital_pct, take_discount}; None when net_days is not after discount_days or an input is absent"),
    notes=(
        "The standard comparison: annualised return = discount / (100 - discount) * 365 / (net_days - discount_days). "
        "take_discount is true when that return beats the cost of capital; whether AP can reliably pay inside the window is the user's statement, not computed here."
    ),
    golden=[
        GoldenVector(inputs={"discount_pct": 2, "discount_days": 10, "net_days": 30, "annual_rate": 0.08}, expected={"annualised_return_pct": 37.24, "cost_of_capital_pct": 8.0, "take_discount": True}, tolerance=0.01),
        GoldenVector(inputs={"discount_pct": 0.5, "discount_days": 10, "net_days": 60, "annual_rate": 0.08}, expected={"annualised_return_pct": 3.67, "cost_of_capital_pct": 8.0, "take_discount": False}, tolerance=0.01),
        GoldenVector(inputs={"discount_pct": 2, "discount_days": 30, "net_days": 30, "annual_rate": 0.08}, expected=None),
    ],
)
def early_payment_decision(discount_pct=None, discount_days=None, net_days=None, annual_rate=None):
    d, dd, nd, r = _num(discount_pct), _num(discount_days), _num(net_days), _num(annual_rate)
    if d is None or dd is None or nd is None or r is None or d < 0 or d >= 100 or nd <= dd or r < 0:
        return None
    annualised = d / (100.0 - d) * 365.0 / (nd - dd) * 100.0
    return {"annualised_return_pct": round(annualised, 2), "cost_of_capital_pct": round(r * 100.0, 2), "take_discount": annualised > r * 100.0}


@formula(
    "finance.net_present_value",
    version="1.0.0",
    owner=_OWNER,
    purpose="Net present value of dated cash flows at an annual rate; year 0 is undiscounted, costs negative, benefits positive",
    effective_from=_FROM,
    inputs=[
        Term("cash_flows", ROWS, "rows of {year, amount}: year 0 today, amount negative for a cost"),
        Term("cost_of_capital", RATIO, "annual discount rate as a fraction, from the governed policy or finance", minimum=0.0),
    ],
    output=Output("float", MONEY, "sum of amount / (1 + rate) ** year, 2dp; None when a row is malformed or the rate is absent"),
    notes=(
        "The rate is an assumption the policy or finance supplies; the argument check refuses one the model made "
        "up. A row without a numeric year and amount makes the whole result None rather than a partial sum."
    ),
    golden=[
        GoldenVector(inputs={"cash_flows": [{"year": 0, "amount": -100000.0}, {"year": 1, "amount": 60000.0}, {"year": 2, "amount": 60000.0}],
                            "cost_of_capital": 0.1}, expected=4132.23, tolerance=0.01),
        GoldenVector(inputs={"cash_flows": [{"year": 0, "amount": -100.0}, {"year": 1, "amount": 100.0}], "cost_of_capital": 0.0}, expected=0.0),
        GoldenVector(inputs={"cash_flows": [{"year": 0, "amount": -100.0}, {"year": "soon", "amount": 100.0}], "cost_of_capital": 0.1}, expected=None),
        GoldenVector(inputs={"cash_flows": [], "cost_of_capital": 0.1}, expected=None),
    ],
)
def net_present_value(cash_flows: Sequence[Mapping[str, Any]] = (), cost_of_capital=None):
    r = _num(cost_of_capital)
    rows = list(cash_flows or [])
    if r is None or r < 0 or not rows:
        return None
    total = 0.0
    for row in rows:
        y, a = _num((row or {}).get("year")), _num((row or {}).get("amount"))
        if y is None or a is None or y < 0:
            return None
        total += a / ((1.0 + r) ** y)
    return round(total, 2)
