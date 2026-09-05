"""Robust price-outlier detection, registered.

Delegates to ``src.services.price_outlier.rule``. Median and MAD rather than
mean and standard deviation, because the mean and the standard deviation are
both dragged by the outliers being hunted --- a single ten-times-wrong price
raises the bar enough to hide itself.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

from services.price_outlier.rule import OutlierSettings, assess as _assess

from ..contract import MONEY, RECORD, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "assurance"
_FROM = date(2026, 9, 5)
_PEERS = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5]


@formula(
    "price_outlier.verdict",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether a unit price is both statistically extreme and commercially material",
    effective_from=_FROM,
    inputs=[
        Term("price", MONEY, "the price under test", minimum=0.0),
        Term("peers", ROWS, "comparable peer prices"),
        Term("settings", RECORD, "OutlierSettings overrides", required=False),
    ],
    output=Output("Verdict", MONEY,
                  "flagged, peer_count, median, ratio, robust_score, severity"),
    notes=(
        "Both tests must pass. Either alone is useless: the statistical test flags "
        "trivia when peers are nearly identical, and the materiality test flags "
        "ordinary price variety. Below `min_peers` the answer is 'not flagged' "
        "with no median -- an abstention, not a clearance."
    ),
    golden=[
        GoldenVector(inputs={"price": 101.0, "peers": _PEERS, "settings": None},
                     expected={"flagged": False, "peer_count": 6, "median": 100.25,
                               "severity": None}),
        GoldenVector(inputs={"price": 1200.0, "peers": _PEERS, "settings": None},
                     expected={"flagged": True, "severity": "critical",
                               "median": 100.25}),
        GoldenVector(inputs={"price": 310.0, "peers": _PEERS, "settings": None},
                     expected={"flagged": True, "severity": "warning"}),
        GoldenVector(inputs={"price": 1200.0, "peers": [100.0, 101.0], "settings": None},
                     expected={"flagged": False, "peer_count": 2, "median": None},
                     note="too few peers -- abstains rather than clearing"),
        GoldenVector(inputs={"price": 1200.0, "peers": [100.0] * 6, "settings": None},
                     expected={"flagged": True, "robust_score": None,
                               "severity": "critical"},
                     note="MAD is zero, so the ratio test stands alone"),
    ],
)
def outlier_verdict(price: float, peers: Sequence[float], settings=None):
    cfg = settings
    if cfg is None:
        cfg = OutlierSettings()
    elif not isinstance(cfg, OutlierSettings):
        cfg = OutlierSettings(**cfg)
    return _assess(float(price), [float(p) for p in peers], cfg)
