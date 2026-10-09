"""Company settings for agent policies, read from proc.bp_admin_config['agent_policy_settings'].

Defaults are the values ruled on 2026-10-08. A missing or unreadable row falls back to them;
it never falls back to something more permissive, because every default here is the safe one.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "response_time": "PT4H",
    "response_time_basis": "clock",
    "on_missing_data": {"approve": "fail_closed", "block": "fail_closed", "notify": "fail_closed"},
    "conflict_cases_per_run": 25,
    "learning": {"min_decisions": 30, "min_days": 30, "min_approvers": 3, "wilson_lower": 0.85,
                 "median_seconds_floor": 30, "not_yet_more": 30, "dismiss_more": 30},
}


def merge(stored: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(DEFAULTS)
    for key, value in (stored or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


def load_settings(conn: Any = None) -> Dict[str, Any]:
    from services.db import get_conn

    try:
        if conn is None:
            with get_conn() as own:
                return load_settings(own)
        cur = conn.cursor()
        cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = 'agent_policy_settings'")
        row = cur.fetchone()
        value = row[0] if row else None
        if isinstance(value, str):
            value = json.loads(value)
        return merge(value)
    except Exception as exc:  # unreadable settings -> ruled defaults, loudly
        logger.warning("agent_policy_settings unreadable, using defaults: %s", exc)
        return merge(None)


#: N, the precedent count (design §3.3): a governed limit in proc.bp_policy that a customer can
#: change, not a company setting. One number for both the precedent and the standing-rule
#: proposal (ruling R2).
PRECEDENT_POLICY = "agent_policy_conflicts"
PRECEDENT_RULE = "precedent_count"


def precedent_count() -> Optional[int]:
    """How many same-way decisions by people make a precedent. Read fresh on every call: the
    policy admin writes the row directly and its edit must apply without a restart. None is a
    stated null (off). Raises governed_limits.LimitUnavailable when the row or rule is missing,
    and ValueError/TypeError when the value is not a number: never a default in code."""
    from src.services import governed_limits

    return governed_limits.limit(PRECEDENT_POLICY, PRECEDENT_RULE, cast=int, fresh=True)


#: How far above the largest value people approved precedent still applies (Task 12): a governed
#: percentage on the same row. 20 = up to 20% above; 0 = never above; null = no range check.
PRECEDENT_RANGE_RULE = "precedent_value_range_pct"


def _percent(value) -> float:
    """float(), except that a JSON true/false is unreadable: never read as 1% or 0%."""
    if isinstance(value, bool):
        raise TypeError("the precedent value range is a boolean, not a number")
    return float(value)


def precedent_value_range_pct() -> Optional[float]:
    """The governed precedent value range, in percent, read fresh like precedent_count. None is a
    stated null (no range check). Raises governed_limits.LimitUnavailable when the row or rule is
    missing, and ValueError/TypeError when the value is not a number (a boolean included) or is
    negative."""
    import math

    from src.services import governed_limits

    pct = governed_limits.limit(PRECEDENT_POLICY, PRECEDENT_RANGE_RULE, cast=_percent, fresh=True)
    if pct is not None and not (math.isfinite(pct) and pct >= 0):
        raise ValueError("the precedent value range must be a number of percent, 0 or more")
    return pct
