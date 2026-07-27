"""The outlier decision, as a pure function.

Median and median-absolute-deviation, not mean and standard deviation. The mean
and the standard deviation are both dragged by the outliers being hunted, so a
single ten-times-wrong price raises the bar enough to hide itself. The median
does not move.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

# Scales the median absolute deviation so that, for normally distributed data,
# it estimates the same spread as the standard deviation.
_MAD_TO_SIGMA = 1.4826

# Peer prices carry ordinary rounding noise (a peer set can median at 100.1
# rather than a clean 100), which must not change whether a 3x or 10x multiple
# is judged material or critical. This is a numerical-noise allowance, not a
# business threshold, so it lives here rather than on OutlierSettings.
_RATIO_TOLERANCE = 0.01


class OutlierSettings(BaseModel):
    """Every threshold injectable; no magic numbers in the rule."""

    model_config = ConfigDict(frozen=True)

    min_peers: int = Field(default=5, ge=2)
    robust_threshold: float = 5.0
    material_ratio: float = Field(default=3.0, gt=1.0)
    critical_ratio: float = Field(default=10.0, gt=1.0)


@dataclass(frozen=True)
class Verdict:
    flagged: bool
    peer_count: int
    median: Optional[float] = None
    ratio: Optional[float] = None
    robust_score: Optional[float] = None
    severity: Optional[str] = None


def _crosses(ratio: float, multiplier: float) -> bool:
    """Is `ratio` at or beyond `multiplier` (or its reciprocal), within the
    noise tolerance? Symmetric so a price ten times too low is judged the
    same as one ten times too high."""
    lo = multiplier * (1.0 - _RATIO_TOLERANCE)
    hi = (1.0 / multiplier) * (1.0 + _RATIO_TOLERANCE)
    return ratio >= lo or ratio <= hi


def assess(
    price: float, peers: Sequence[float], settings: OutlierSettings
) -> Verdict:
    """Is `price` extreme against `peers`?

    Flags only when the price is BOTH statistically extreme and commercially
    material. Either test alone is useless: the first flags trivia when peers
    are nearly identical, the second flags ordinary price variety.
    """
    n = len(peers)
    if n < settings.min_peers:
        return Verdict(flagged=False, peer_count=n)

    mid = float(median(peers))
    if mid <= 0:
        # No ratio can be formed against a zero or negative median.
        return Verdict(flagged=False, peer_count=n, median=mid)

    ratio = price / mid
    material = _crosses(ratio, settings.material_ratio)

    mad = float(median([abs(p - mid) for p in peers])) * _MAD_TO_SIGMA
    if mad == 0:
        # More than half the peers share one price, so deviation is undefined.
        # The ratio test stands alone rather than flagging everything.
        robust = None
        extreme = material
    else:
        robust = abs(price - mid) / mad
        extreme = robust >= settings.robust_threshold

    flagged = material and extreme
    severity = None
    if flagged:
        severity = "critical" if _crosses(ratio, settings.critical_ratio) else "warning"

    return Verdict(
        flagged=flagged, peer_count=n, median=mid, ratio=ratio,
        robust_score=robust, severity=severity,
    )
