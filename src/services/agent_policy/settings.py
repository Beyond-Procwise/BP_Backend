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
    "live_conflict_repeat": 5,
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
