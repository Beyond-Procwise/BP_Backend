"""Weighted scoring of supplier evaluations, registered.

Delegates to ``src.services.scoring.weighted``. The ranking is arithmetic over scores and weights
a person supplied; it never picks the winner and never changes an input.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Mapping, Optional, Sequence

from src.services.scoring import weighted as _w

from ..contract import COUNT, FACTOR, LABEL, MONEY, ROWS, Output, Term
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED, decline

log = logging.getLogger(__name__)

_CRITERIA = [
    {"criterion_id": "quality", "parent_id": None, "weight": "0.6", "type": "scored"},
    {"criterion_id": "price", "parent_id": None, "weight": "0.4", "type": "price", "direction": "lower_better"},
    {"criterion_id": "security", "parent_id": None, "weight": "0", "type": "gate_pass_fail"},
]
_SCORES = [
    {"supplier_id": "A", "criterion_id": "quality", "evaluator_id": "e1", "score": 4},
    {"supplier_id": "B", "criterion_id": "quality", "evaluator_id": "e1", "score": 3},
    {"supplier_id": "A", "criterion_id": "security", "evaluator_id": "e1", "score": 1},
    {"supplier_id": "B", "criterion_id": "security", "evaluator_id": "e1", "score": 1},
]
_METRICS = [{"supplier_id": "A", "criterion_id": "price", "value": 100},
            {"supplier_id": "B", "criterion_id": "price", "value": 120}]
_CASE = dict(criteria=_CRITERIA, scale_min=0, scale_max=5, scores=_SCORES, metrics=_METRICS,
             normalisation="ratio_to_best")


@formula(
    "scoring.weighted_total",
    version="1.0.0",
    owner="scoring",
    purpose="Ranked supplier totals from evaluator scores and criterion weights, with gates, a named "
            "normalisation, consistency flags and weight sensitivity",
    effective_from=date(2026, 10, 21),
    inputs=[
        Term("criteria", ROWS, "rows like {criterion_id, parent_id (null at top level), weight (decimal string), "
             "type: scored | price | numeric_metric | gate_pass_fail | group, direction: lower_better | higher_better}"),
        Term("scale_min", FACTOR, "lowest score an evaluator may give"),
        Term("scale_max", FACTOR, "highest score an evaluator may give"),
        Term("scores", ROWS, "rows like {supplier_id, criterion_id, evaluator_id, score}; evaluator_id may be left out on every row when one person scored; gates: 1 pass, 0 fail"),
        Term("metrics", ROWS, "rows like {supplier_id, criterion_id, value} for price and numeric criteria",
             required=False),
        Term("consensus_scores", ROWS, "rows like {supplier_id, criterion_id, score} agreed by the panel",
             required=False),
        Term("normalisation", LABEL, "how price and numeric criteria become 0-1", required=False,
             allowed=_w.NORMALISATIONS),
        Term("aggregation", LABEL, "how evaluator scores combine", required=False,
             allowed=_w.AGGREGATIONS),
        Term("price_reference", MONEY, "the price that scores 0 under price_interpolation",
             required=False),
        Term("monte_carlo_n", COUNT, "weight-perturbation draws; 0 = none", minimum=0.0, required=False),
        Term("seed", COUNT, "required when monte_carlo_n is above 0", required=False),
    ],
    output=Output("dict", FACTOR,
                  "ranking (total, rank, tied, contributions), excluded, global_weights, "
                  "aggregated_scores, evaluator_flags, criterion_flags, sensitivity, warnings, "
                  "not_computed, top_supplier, top_total; UNASSESSED if the inputs cannot be scored"),
    notes=(
        "Weights at each sibling level must sum to 1 (decimal arithmetic, tolerance 1e-9); gates have no "
        "weight and a failed gate removes the supplier. Scored criteria are normalised on the scale; price "
        "and numeric criteria use the named method. Global weight is the product down the tree. Equal totals "
        "to six decimals are a tie with a shared rank: nothing breaks a tie quietly. A missing score is "
        "refused, never counted as zero. Krippendorff alpha, ICC and evaluator rank-reversal checks are NOT "
        "computed and are listed in not_computed. It ranks; it never chooses the winner."
    ),
    golden=[
        GoldenVector(
            inputs=_CASE,
            expected={"top_supplier": "A", "top_total": 0.88},
            tolerance=1e-9,
            note="quality 0.6 (A 4/5, B 3/5), price 0.4 ratio-to-best (A 100, B 120): A = 0.48 + 0.40",
        ),
        GoldenVector(
            inputs={**_CASE, "criteria": [dict(_CRITERIA[0], weight="0.6"), dict(_CRITERIA[1], weight="0.3"),
                                          _CRITERIA[2]]},
            expected=UNASSESSED,
            note="weights sum to 0.9: nothing is rescaled or guessed",
        ),
    ],
)
def scoring_weighted_total(
    criteria: Sequence[Mapping[str, Any]], scale_min: float, scale_max: float,
    scores: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]] = (),
    consensus_scores: Sequence[Mapping[str, Any]] = (), normalisation: str = "ratio_to_best",
    aggregation: str = "consensus_if_present", price_reference: Optional[float] = None,
    monte_carlo_n: int = 0, seed: Optional[int] = None,
):
    try:
        out = _w.weighted_scoring(
            criteria=criteria, scale_min=scale_min, scale_max=scale_max, scores=scores,
            consensus_scores=consensus_scores or (), metrics=metrics or (),
            normalisation=normalisation or "ratio_to_best", aggregation=aggregation or "consensus_if_present",
            price_reference=price_reference, monte_carlo_n=int(monte_carlo_n or 0),
            seed=None if seed is None else int(seed))
    except _w.ScoringError as exc:
        log.info("scoring.weighted_total could not score: %s", exc)
        return decline(str(exc))
    top = out["ranking"][0] if out["ranking"] else None
    out["top_supplier"] = top["supplier_id"] if top else None
    out["top_total"] = top["total"] if top else None
    return out
