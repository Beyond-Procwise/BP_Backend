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

from typing import Any, Dict, List, Optional, Tuple

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


# --- the Quote builder's settings (ruled 2026-10-14) -----------------------------------
# Company defaults, in the same governed row as the floor. Each is read fresh, so an
# edit in the policy admin applies on the next read. `locked`: they are governed, so the
# Quote builder shows them and does not edit them.

def _days(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 < value <= 3650 else None


def _words(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:300]


_SETTINGS = (("quote.tax.rate_pct", "tax_rate_pct", _percent),
             ("quote.validity.days", "validity_days", _days),
             ("quote.terms.payment_default", "payment_terms_default", _words),
             ("quote.margin.floor_pct", "margin_floor_pct", _percent))


def settings() -> Dict[str, Dict[str, Any]]:
    """{key: {value, source, locked}}. A null value is UNSET: the screen and the server
    both refuse what needs it, and neither reads it as zero."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, rule, usable in _SETTINGS:
        stated, stated_as = _stated(rule)
        value = usable(stated)
        out[key] = {"value": value, "source": _source(value, stated_as), "locked": True}
    return out


def default_tax_rate() -> Optional[float]:
    """The rate a new quote is priced at, or None while nobody has set one."""
    stated, _ = _stated("tax_rate_pct")
    return _percent(stated)


def unset_blocks(quote_tax_rate_pct: Any) -> List[Dict[str, str]]:
    """What stops a builder quote going for review or being sent: a margin floor nobody
    has set, and a quote with no tax rate. Empty when neither applies."""
    floor, _ = _stated("margin_floor_pct")
    blocks = []
    if _percent(floor) is None:
        blocks.append({"code": "MARGIN_POLICY_UNSET",
                       "message": "No margin floor is set in pricing policy."})
    if quote_tax_rate_pct is None:
        blocks.append({"code": "TAX_POLICY_UNSET",
                       "message": "The quote has no tax rate; none is set in pricing policy."})
    return blocks
