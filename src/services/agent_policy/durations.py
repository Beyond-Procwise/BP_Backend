"""The one ISO 8601 duration reader for agent policies.

Used by the enforcement check (what wait the agent is told) and by the approval timer (when a
case escalates or times out), so the two can never disagree.

Accepted: PnW, and PnDTnHnMnS in any subset, with fractional components ("PT1.5H", "P0,5D").
Anything else, or a zero length, is unreadable and resolves to the company default.
"""
from __future__ import annotations

import re
from typing import Any, Optional

COMPANY_DEFAULT = "PT4H"

_NUM = r"(\d+(?:[.,]\d+)?)"
_DURATION = re.compile(
    rf"^P(?!$)(?:{_NUM}W)?(?:{_NUM}D)?(?:T(?=\d)(?:{_NUM}H)?(?:{_NUM}M)?(?:{_NUM}S)?)?$")


def parse(value: Any) -> Optional[float]:
    """Seconds in the duration, or None when it is unreadable or zero."""
    if not isinstance(value, str):
        return None
    m = _DURATION.match(value.strip().upper())
    if not m:
        return None
    w, d, h, mi, s = (float((x or "0").replace(",", ".")) for x in m.groups())
    secs = (((w * 7 + d) * 24 + h) * 60 + mi) * 60 + s
    return secs if secs > 0 else None


def resolve(value: Any, default: Any = COMPANY_DEFAULT) -> str:
    """The ISO string to apply: the value when readable, otherwise the default (itself checked)."""
    if parse(value) is not None:
        return value.strip().upper()
    if parse(default) is not None:
        return default.strip().upper()
    return COMPANY_DEFAULT
