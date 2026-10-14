"""Supplier ranking and quote comparison --- the population-scoped formulas.

These are the reason ``FormulaKind.SET`` exists. A min-max normalised criterion
score and a ratio-to-cheapest price score are properties of the *frame*, not of
a row: evaluate one supplier in isolation and you cannot reproduce either
number. Registering them as SET formulas records that fact in the type system,
so a later "optimisation" into a per-row loop fails loudly instead of quietly
changing every ranking the system has ever produced.

Every body delegates to the live implementation in ``supplier_ranking_agent`` or
``quote_comparison_agent``. NaN is preserved exactly as it is today: it means
"we hold no measurement for this supplier", which is not the same claim as zero.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.agents.quote_comparison_agent import QuoteComparisonAgent as _QCA
from src.agents.supplier_ranking_agent import SupplierRankingAgent as _SRA
from src.agents.supplier_ranking_agent import (
    _normalize_days_to_score as _pay_terms,
    composite_scores as _composite,
)

from ..contract import (
    COUNT, DAYS, LABEL, MONEY, RATIO, RECORD, ROWS, SCORE_100, Output, Term,
)
from ..registry import FormulaKind, GoldenVector, formula

_OWNER = "sourcing"
_FROM = date(2026, 9, 5)
_NAN = float("nan")


@formula(
    "supplier.payment_terms_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="Payment terms in days scored 0-100, longer terms being better for the buyer",
    effective_from=_FROM,
    inputs=[
        Term("payment_terms_days", DAYS, "credit days offered",
             minimum=0.0, required=False, concept_code="payment_terms"),
        Term("min_days", DAYS, "bottom of the scale", minimum=0.0, required=False),
        Term("max_days", DAYS, "top of the scale", minimum=0.0, required=False),
    ],
    output=Output("float", SCORE_100, "0-100, 2dp; None when no terms are on file"),
    notes=(
        "Returns None, not 0.0, for a supplier with no terms recorded. Zero would "
        "assert 'they offered the worst terms available', which is a claim about "
        "the supplier rather than about our data."
    ),
    golden=[
        GoldenVector(inputs={"payment_terms_days": 0, "min_days": None, "max_days": None},
                     expected=100.0, note="due on receipt is the best possible for a buyer"),
        GoldenVector(inputs={"payment_terms_days": 30, "min_days": None, "max_days": None},
                     expected=66.67),
        GoldenVector(inputs={"payment_terms_days": 45, "min_days": None, "max_days": None},
                     expected=50.0),
        GoldenVector(inputs={"payment_terms_days": 90, "min_days": None, "max_days": None},
                     expected=0.0),
        GoldenVector(inputs={"payment_terms_days": 120, "min_days": None, "max_days": None},
                     expected=0.0, note="clamped, not extrapolated below zero"),
        GoldenVector(inputs={"payment_terms_days": None, "min_days": None, "max_days": None},
                     expected=None, note="unknown terms stay unknown"),
    ],
)
def payment_terms_score(payment_terms_days=None, min_days=None, max_days=None):
    if min_days is None and max_days is None:
        return _pay_terms(payment_terms_days)
    return _pay_terms(
        payment_terms_days,
        0.0 if min_days is None else float(min_days),
        90.0 if max_days is None else float(max_days),
    )


@formula(
    "supplier.deal_price_score",
    version="1.0.0",
    kind=FormulaKind.SET,
    owner=_OWNER,
    purpose="Price scored against the cheapest RIVAL bid on the same deal (100 = cheapest)",
    effective_from=_FROM,
    inputs=[Term("price", MONEY, "this supplier's price on this deal",
                 minimum=0.0, required=False)],
    output=Output("list[float]", SCORE_100,
                  "100 x cheapest / price, 2dp; NaN for every bid when fewer than two bid"),
    notes=(
        "Ratio-to-best, not min-max. Min-max stretched whatever gap existed across the "
        "full 0-100 scale: on the live TEST005 deal it scored a bid 1.1% more expensive "
        "as 0.00 against the winner's 100.00, when all three bids sat within GBP 3,050 "
        "of each other. The ordering was right and the numbers slandered the runners-up. "
        "A lone bidder scores NaN, not 100 -- an uncontested quote is no evidence of a "
        "good price."
    ),
    golden=[
        GoldenVector(
            inputs=[{"price": 100000.0}, {"price": 101100.0}, {"price": 103050.0}],
            expected=[100.0, 98.91, 97.04],
            note="the TEST005 shape: 98.91 means '1.1% off the best price'",
        ),
        GoldenVector(inputs=[{"price": 100000.0}], expected=[_NAN],
                     note="one bidder is not a competition"),
        GoldenVector(inputs=[{"price": 0.0}, {"price": 50.0}], expected=[_NAN, _NAN],
                     note="a zero cheapest bid cannot form a ratio"),
    ],
)
def deal_price_score(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    df = pd.DataFrame({"price": [r.get("price") for r in rows]})
    out = _SRA._score_deal_price(None, df)
    return [float(v) for v in out["price_score"]]


@formula(
    "supplier.criterion_normalisation",
    version="1.0.0",
    kind=FormulaKind.SET,
    owner=_OWNER,
    purpose="Min-max normalise one raw criterion to 0-100 across the suppliers being compared",
    effective_from=_FROM,
    inputs=[
        Term("value", RATIO, "the raw criterion value for this supplier", required=False),
        Term("direction", LABEL, "lower_is_better or higher_is_better",
             allowed=("lower_is_better", "higher_is_better"), shared=True),
    ],
    output=Output("list[float]", SCORE_100, "0-100 per supplier; NaN where unmeasured"),
    notes=(
        "All-unmeasured yields NaN for everyone, so `supplier.weight_renormalisation` "
        "drops the criterion rather than dragging every composite down by its weight. "
        "When every MEASURED supplier ties, only the measured ones score 100 -- "
        "assigning the tied score to the whole column once handed an unmeasured "
        "supplier a perfect 100 they never earned."
    ),
    golden=[
        GoldenVector(inputs=[{"value": 5.0}, {"value": 10.0}, {"value": 20.0}],
                     shared={"direction": "lower_is_better"},
                     expected=[100.0, 66.66666666666667, 0.0]),
        GoldenVector(inputs=[{"value": 5.0}, {"value": 10.0}, {"value": 20.0}],
                     shared={"direction": "higher_is_better"},
                     expected=[0.0, 33.333333333333336, 100.0]),
        GoldenVector(inputs=[{"value": None}, {"value": None}, {"value": None}],
                     shared={"direction": "higher_is_better"},
                     expected=[_NAN, _NAN, _NAN], note="unmeasured, not worst"),
        GoldenVector(inputs=[{"value": 7.0}, {"value": 7.0}, {"value": None}],
                     shared={"direction": "higher_is_better"},
                     expected=[100.0, 100.0, _NAN],
                     note="a tie among the measured does not enrol the unmeasured"),
    ],
)
def criterion_normalisation(rows: Sequence[Mapping[str, Any]], direction: str) -> List[float]:
    df = pd.DataFrame({"c": [r.get("value") for r in rows]})
    out = _SRA._normalize_numeric_scores(None, df, {"c": direction})
    return [float(v) for v in out["c_score"]]


@formula(
    "supplier.weight_renormalisation",
    version="1.0.0",
    owner=_OWNER,
    purpose="Drop criteria nobody can be scored on, and renormalise the rest to sum to 1",
    effective_from=_FROM,
    inputs=[
        Term("weights", RECORD, "criterion -> weight, as configured or governed"),
        Term("scored", RECORD,
             "criterion -> list of that criterion's scores across the population"),
    ],
    output=Output("dict", RATIO, "criterion -> normalised weight, summing to 1.0"),
    notes=(
        "A criterion only counts if a real `_score` column exists AND is not entirely "
        "NaN. Accepting one on the strength of its raw column let weights and scoring "
        "disagree: the weight map kept the criterion, the scoring loop skipped it for "
        "want of a score column, and its weight silently vanished from the composite."
    ),
    golden=[
        GoldenVector(
            inputs={"weights": {"price": 0.5, "delivery": 0.3, "risk": 0.2},
                    "scored": {"price": [1.0, 2.0], "delivery": [_NAN, _NAN]}},
            expected={"price": 1.0},
            note="delivery is all-NaN and risk has no column; price absorbs the weight",
        ),
        GoldenVector(
            inputs={"weights": {"delivery": 0.3},
                    "scored": {"price": [1.0, 2.0], "delivery": [_NAN, _NAN]}},
            expected={},
            note="nothing scoreable -> empty, so the caller keeps its own weights",
        ),
    ],
)
def weight_renormalisation(weights: Mapping[str, float],
                           scored: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    df = pd.DataFrame({f"{k}_score": list(v) for k, v in (scored or {}).items()})
    return _SRA._normalise_weight_map(None, df, dict(weights or {}))


@formula(
    "supplier.composite_score",
    version="1.0.0",
    kind=FormulaKind.SET,
    owner=_OWNER,
    purpose="Weighted composite supplier score over only the criteria we hold for them",
    effective_from=_FROM,
    inputs=[
        Term("scores", RECORD, "criterion -> 0-100 score for this supplier, NaN if unheld"),
        Term("weights", RECORD, "criterion -> weight, shared across the population",
             shared=True),
    ],
    output=Output("list[float]", SCORE_100,
                  "composite per supplier; NaN when no criterion was measurable"),
    notes=(
        "Each supplier's weights are renormalised over the criteria they actually have, "
        "so a gap in OUR data is not charged to THEIR bid. A supplier with nothing "
        "measurable scores NaN rather than 0.0 -- we have no opinion, and 0 would be "
        "inventing one. Compare `quote.weighting_score`, which returns 0.0 for the "
        "same situation (gap report D-3)."
    ),
    golden=[
        GoldenVector(
            inputs=[
                {"scores": {"price": 100.0, "delivery": 80.0, "payment_terms": 66.67}},
                {"scores": {"price": 98.9, "delivery": 90.0, "payment_terms": _NAN}},
                {"scores": {"price": _NAN, "delivery": _NAN, "payment_terms": _NAN}},
            ],
            shared={"weights": {"price": 0.5, "delivery": 0.3, "payment_terms": 0.2}},
            expected=[87.334, 95.5625, _NAN],
            note=(
                "supplier B is missing payment terms and is scored on the other two "
                "renormalised, NOT charged 0 for the gap"
            ),
        ),
        GoldenVector(
            inputs=[{"scores": {"price": 100.0}}, {"scores": {"price": 50.0}}],
            shared={"weights": {"quality": 1.0}},
            expected=[_NAN, _NAN],
            note="no weighted criterion has a score column -> no opinion at all",
        ),
    ],
)
def composite_score(rows: Sequence[Mapping[str, Any]],
                    weights: Mapping[str, float]) -> List[float]:
    criteria = sorted({c for r in rows for c in (r.get("scores") or {})})
    frame = pd.DataFrame(
        {f"{c}_score": [(r.get("scores") or {}).get(c, _NAN) for r in rows]
         for c in criteria}
    )
    if frame.empty:
        frame = pd.DataFrame(index=range(len(rows)))
    final, _scored_on = _composite(frame, dict(weights or {}))
    return [float(v) for v in final]


@formula(
    "quote.weighting_score",
    version="1.0.0",
    kind=FormulaKind.SET,
    owner=_OWNER,
    purpose="Weighted 0-100 comparison score across competing quotes",
    effective_from=_FROM,
    inputs=[
        Term("total_cost_gbp", MONEY, "landed total cost", minimum=0.0, required=False),
        Term("tenure", COUNT, "lead time (lower is better, despite the name)",
             minimum=0.0, required=False),
        Term("volume", COUNT, "quoted volume", minimum=0.0, required=False),
        Term("weights", RECORD, "metric -> weight; empty falls back to the defaults",
             shared=True, required=False),
    ],
    output=Output("list[float]", SCORE_100, "0-100 per quote"),
    notes=(
        "**Duplicates `supplier.composite_score` and disagrees with it (gap report "
        "D-3).** Both min-max normalise then take a weighted mean; this one scores a "
        "quote with no usable metric as **0.0**, which ranks it as the worst on the "
        "table, while supplier ranking returns NaN for the identical situation and "
        "documents why. Behaviour unchanged and pinned by the last vector, so a "
        "consolidation is a deliberate version bump with visible numbers."
    ),
    golden=[
        GoldenVector(
            inputs=[
                {"total_cost_gbp": 100000.0, "tenure": 10.0, "volume": 500.0,
                 "weights": None},
                {"total_cost_gbp": 110000.0, "tenure": 5.0, "volume": 800.0,
                 "weights": None},
                {"total_cost_gbp": 105000.0, "tenure": 8.0, "volume": 650.0,
                 "weights": None},
            ],
            shared={"weights": None},
            expected=[60.0, 40.0, 47.5],
            note="default weights 0.6 / 0.25 / 0.15",
        ),
        GoldenVector(
            inputs=[{"total_cost_gbp": None, "tenure": None, "volume": None,
                     "weights": None},
                    {"total_cost_gbp": None, "tenure": None, "volume": None,
                     "weights": None}],
            shared={"weights": None},
            expected=[0.0, 0.0],
            note="PINS D-3: no data scores 0.0, i.e. joint worst",
        ),
    ],
)
def quote_weighting_score(rows: Sequence[Mapping[str, Any]],
                          weights: Optional[Mapping[str, float]] = None) -> List[float]:
    agent = _QCA.__new__(_QCA)
    agent._resolved_metric_weights = {}
    entries: List[Dict[str, Any]] = [
        {"total_cost_gbp": r.get("total_cost_gbp"),
         "tenure": r.get("tenure"),
         "volume": r.get("volume")}
        for r in rows
    ]
    agent._calculate_weighting_scores(entries, dict(weights or {}))
    return [float(e.get("weighting_score", 0.0)) for e in entries]
