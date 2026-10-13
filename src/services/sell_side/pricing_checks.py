"""What a quote is tested against: the margin floor, the discount corridor, cost-to-serve.

Each is a pricing-policy fact and each can be NOT SET. Unset is reported as None with
the reason beside it and is never read as zero: a floor of 0 passes every quote and a
corridor of 0 fails every discount, and neither is what "nobody has decided" means.

The floor and the corridor are governed limits (bp_policy `sales_quote_pricing`).
Cost-to-serve has no model anywhere in the platform, so it is always None here and the
margin rests on the cost the seller entered; the day a model exists it is offered beside
the entered figure, not in place of it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.services import governed_limits

POLICY = "sales_quote_pricing"
SET, NOT_SET, UNAVAILABLE = "policy", "not_set", "unavailable"


def _stated(rule: str) -> Tuple[Any, str]:
    try:
        value = governed_limits.limit(POLICY, rule, cast=lambda v: v, fresh=True)
    except governed_limits.LimitUnavailable:
        return None, UNAVAILABLE
    return (value, SET) if value is not None else (None, NOT_SET)


def _percent(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 <= value <= 100 else None


def _corridor(stated: Any, category: Optional[str]) -> Optional[float]:
    """One number for every category, or a number per category with a "default"."""
    if isinstance(stated, dict):
        return _percent(stated.get(category) if category in stated else stated.get("default"))
    return _percent(stated)


def _source(value: Any, stated_as: str) -> str:
    # a stated limit that gives this quote no usable figure (wrong type, out of range, or
    # no entry for its category) is "not set" for it, not a limit of nothing
    return NOT_SET if value is None and stated_as == SET else stated_as


def for_quote(category: Optional[str]) -> Dict[str, Any]:
    floor, floor_source = _stated("margin_floor_pct")
    corridor, corridor_source = _stated("discount_corridor_max_pct")
    floor, max_pct = _percent(floor), _corridor(corridor, category)
    return {
        "costToServe": None,
        "discountCorridor": {"maxPct": max_pct} if max_pct is not None else None,
        "marginFloorPct": floor,
        # why a figure is what it is: 'policy' (a stated limit), 'not_set', or
        # 'unavailable' (the limits could not be read at all)
        "sources": {"costToServe": "not_modelled",
                    "discountCorridor": _source(max_pct, corridor_source),
                    "marginFloorPct": _source(floor, floor_source)},
    }
