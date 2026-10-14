"""Opportunity detection maths, registered.

The 14 detectors in ``opportunity_miner_agent`` each compute a financial impact
and then compete for attention through a single weighting. Both halves are here:
the shared weighting, and the risk normalisation that feeds it.
"""
from __future__ import annotations

from datetime import date
from typing import Any, List, Mapping, Sequence

from src.agents.opportunity_miner_agent import (
    OpportunityMinerAgent as _OMA,
    finding_weight_factor as _factor,
    normalise_weightages as _normalise,
)

from ..contract import COUNT, GBP, MONEY, RATIO, RECORD, ROWS, Output, Term
from ..registry import FormulaKind, GoldenVector, formula

_OWNER = "opportunity"
_FROM = date(2026, 9, 5)


@formula(
    "opportunity.risk_normalisation",
    version="1.0.0",
    owner=_OWNER,
    purpose="Coerce a supplier risk value onto 0-1, whatever scale it arrived on",
    effective_from=_FROM,
    inputs=[Term("value", RATIO, "risk as stored -- 0-1 or 0-100", required=False)],
    output=Output("float", RATIO, "0-1"),
    notes=(
        "**Fails open (gap report D-4).** An unparseable value returns 0.0, i.e. NO "
        "risk. `risk_score` is stored as VARCHAR in proc.bp_supplier, so a malformed "
        "value is a live possibility, and it currently reads as the safest supplier on "
        "the table before multiplying into the finding weightage. Behaviour unchanged "
        "and pinned by the fourth vector."
    ),
    golden=[
        GoldenVector(inputs={"value": 0.5}, expected=0.5),
        GoldenVector(inputs={"value": 75.0}, expected=0.75,
                     note="anything above 1 is assumed to be a percentage"),
        GoldenVector(inputs={"value": -3.0}, expected=0.0),
        GoldenVector(inputs={"value": "abc"}, expected=0.0,
                     note="PINS D-4: unparseable becomes zero risk, not unknown risk"),
        GoldenVector(inputs={"value": 250.0}, expected=1.0),
    ],
)
def risk_normalisation(value=None) -> float:
    return _OMA._normalise_risk_score(None, value)


@formula(
    "opportunity.finding_weight_factor",
    version="1.0.0",
    owner=_OWNER,
    purpose="One finding's unnormalised weight: money at stake, amplified by risk and coverage",
    effective_from=_FROM,
    inputs=[
        Term("base_impact", GBP, "financial impact, floored at zero", minimum=0.0),
        Term("risk_score", RATIO, "normalised supplier risk", minimum=0.0, maximum=1.0),
        Term("coverage", RATIO, "how much of this supplier's document flow we can see",
             minimum=0.0, maximum=1.0),
    ],
    output=Output("float", GBP, "unnormalised weight factor"),
    notes=(
        "The guard is load-bearing: a finding with real money behind it must never fall "
        "to a zero weight because risk or coverage arrived out of range, or it would "
        "vanish from the ranking entirely -- the opposite of what a risk signal should do."
    ),
    golden=[
        GoldenVector(inputs={"base_impact": 1000.0, "risk_score": 0.5, "coverage": 0.25},
                     expected=1875.0),
        GoldenVector(inputs={"base_impact": 0.0, "risk_score": 0.5, "coverage": 0.25},
                     expected=0.0, note="no money at stake, no weight"),
    ],
)
def finding_weight_factor(base_impact: float, risk_score: float, coverage: float) -> float:
    return _factor(float(base_impact), float(risk_score), float(coverage))


@formula(
    "opportunity.weightage_shares",
    version="1.0.0",
    kind=FormulaKind.SET,
    owner=_OWNER,
    purpose="Turn a run's weight factors into shares of that run",
    effective_from=_FROM,
    inputs=[Term("factor", GBP, "this finding's unnormalised weight", minimum=0.0)],
    output=Output("list[float]", RATIO, "shares summing to 1.0, or all zeros"),
    notes=(
        "Population-scoped: a share only means anything relative to the rest of the "
        "run. A zero total yields zeros rather than an even 1/n split, which would "
        "invent a ranking the evidence does not support."
    ),
    golden=[
        GoldenVector(inputs=[{"factor": 3.0}, {"factor": 1.0}], expected=[0.75, 0.25]),
        GoldenVector(inputs=[{"factor": 0.0}, {"factor": 0.0}], expected=[0.0, 0.0]),
    ],
)
def weightage_shares(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    return _normalise([float(r.get("factor") or 0.0) for r in rows])


@formula(
    "opportunity.price_variance_impact",
    version="1.0.0",
    owner=_OWNER,
    purpose="Money at stake when a paid price sits above a benchmark",
    effective_from=_FROM,
    inputs=[
        Term("actual_price", MONEY, "the price actually paid or quoted", minimum=0.0),
        Term("benchmark_price", MONEY, "the benchmark it is measured against", minimum=0.0),
        Term("quantity", COUNT, "units the gap applies to", minimum=0.0),
    ],
    output=Output("float", GBP, "positive = overpaying against the benchmark"),
    notes=(
        "The shape `(actual - benchmark) x quantity` recurs across several detectors "
        "(price benchmark variance, invoice-vs-PO variance, cheapest-alternative). "
        "Registered once so the shape has a name; the detectors are not yet rewired "
        "onto it, because each carries its own guards and rewiring them is a separate "
        "change with its own before/after numbers."
    ),
    golden=[
        GoldenVector(inputs={"actual_price": 120.0, "benchmark_price": 100.0,
                             "quantity": 250.0}, expected=5000.0),
        GoldenVector(inputs={"actual_price": 90.0, "benchmark_price": 100.0,
                             "quantity": 250.0}, expected=-2500.0,
                     note="a good price is a negative impact, not a clamp to zero"),
    ],
)
def price_variance_impact(actual_price: float, benchmark_price: float,
                          quantity: float) -> float:
    return (float(actual_price) - float(benchmark_price)) * float(quantity)
