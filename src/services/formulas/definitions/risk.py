"""Predictive supplier risk, registered --- with the clock made an input.

The model blends exponentially time-decayed signal severity with a weighted
performance term and squashes the result through a logistic. It read
``datetime.now()`` internally, which meant the same supplier and the same
signals produced a different score a week later with nothing having changed:
not reproducible, not back-testable, and a stored score could not be recomputed
to check it. ``as_of`` is now an explicit parameter defaulting to now, so every
existing caller gets the identical number and an auditable evaluation can pin
the time it was measured from.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from services.risk_intelligence_service import PredictiveRiskModel

from ..contract import HOURS, RATIO, RECORD, ROWS, TIMESTAMP, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "risk"
_FROM = date(2026, 9, 5)
_MODEL = PredictiveRiskModel()


@formula(
    "risk.predictive_supplier_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="Forward-looking supplier risk 0-1 from decayed incident signals and performance",
    effective_from=_FROM,
    inputs=[
        Term("supplier_metrics", RECORD,
             "on_time_delivery_rate, quality_score, anomaly_index, resilience_index"),
        Term("signals", ROWS, "SupplierRiskSignal records", required=False),
        Term("as_of", TIMESTAMP,
             "the point in time signal decay is measured from", required=False),
        Term("half_life_hours", HOURS, "signal half-life", minimum=1.0, required=False),
    ],
    output=Output("dict", RATIO,
                  "score plus its signal and performance components"),
    notes=(
        "**Fails open on missing data (gap report D-2).** An absent "
        "`on_time_delivery_rate` or `quality_score` defaults to 1.0 and an absent "
        "`anomaly_index` to 0.0, so a supplier we hold NO performance data for scores "
        "as a perfect performer. On a risk model that is the wrong direction: absence "
        "of evidence becomes evidence of safety. Behaviour is unchanged here and "
        "pinned by the second vector below, so a fix is a visible version bump."
    ),
    golden=[
        GoldenVector(
            inputs={"supplier_metrics": {"on_time_delivery_rate": 0.92,
                                         "quality_score": 0.88,
                                         "anomaly_index": 0.1,
                                         "resilience_index": 0.7},
                    "signals": [], "as_of": None, "half_life_hours": None},
            expected={"score": 0.027151396260212884,
                      "performance_component": 0.11699999999999999,
                      "signal_component": 0.0, "signals_considered": 0.0},
            note="no signals, full metrics",
        ),
        GoldenVector(
            inputs={"supplier_metrics": {}, "signals": [], "as_of": None,
                    "half_life_hours": None},
            expected={"score": 0.021457289717603523,
                      "performance_component": 0.05},
            note=(
                "PINS D-2: a supplier with NO metrics at all scores 0.021 -- safer "
                "than the fully-measured supplier above. Deliberately recorded, "
                "not fixed."
            ),
        ),
    ],
)
def predictive_supplier_score(supplier_metrics, signals=None, as_of=None,
                              half_life_hours=None):
    model = (
        _MODEL if half_life_hours is None
        else PredictiveRiskModel(half_life_hours=float(half_life_hours))
    )
    return model.evaluate(dict(supplier_metrics or {}), list(signals or []), as_of=as_of)
