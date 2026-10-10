"""Company settings for agent policies, read from proc.bp_admin_config['agent_policy_settings'].

Defaults are the values ruled on 2026-10-08. A missing or unreadable row falls back to them;
it never falls back to something more permissive, because every default here is the safe one.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "response_time": "PT4H",
    "response_time_basis": "clock",
    "on_missing_data": {"approve": "fail_closed", "block": "fail_closed", "notify": "fail_closed"},
    "conflict_cases_per_run": 25,
    "learning": {"min_decisions": 30, "min_days": 30, "min_approvers": 3, "wilson_lower": 0.85,
                 "median_seconds_floor": 30, "not_yet_more": 30, "dismiss_more": 30},
}


#: The row of proc.bp_admin_config these settings live in.
CONFIG_KEY = "agent_policy_settings"

#: The learning thresholds an administrator may change, in the order they are shown.
LEARNING_KEYS: Tuple[str, ...] = tuple(DEFAULTS["learning"])
#: What each may be: (whole number only?, lowest, highest), both ends allowed.
LEARNING_RANGES: Dict[str, Tuple[bool, float, float]] = {
    "min_decisions": (True, 10, 1000),
    "min_days": (True, 1, 365),
    "min_approvers": (True, 2, 50),
    "wilson_lower": (False, 0.5, 0.99),
    "median_seconds_floor": (True, 0, 3600),
    "not_yet_more": (True, 1, 1000),
    "dismiss_more": (True, 1, 1000),
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
        cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = %s", (CONFIG_KEY,))
        row = cur.fetchone()
        value = row[0] if row else None
        if isinstance(value, str):
            value = json.loads(value)
        return merge(value)
    except Exception as exc:  # unreadable settings -> ruled defaults, loudly
        logger.warning("agent_policy_settings unreadable, using defaults: %s", exc)
        return merge(None)


# ---------------------------------------------------------------- the learning thresholds (stage 5b)

def learning_scan_enabled() -> bool:
    """Whether the hourly learning scan is switched on in this process: AGENT_POLICY_LEARNING_SCAN
    is on, 1 or true (any case, spaces trimmed). Unset, and every other value, is off. The
    scheduler registers the scan on this answer and the settings screen shows it."""
    return os.environ.get("AGENT_POLICY_LEARNING_SCAN", "").strip().lower() in ("on", "1", "true")


def _in_range(key: str, value: Any) -> bool:
    whole, low, high = LEARNING_RANGES[key]
    if isinstance(value, bool):                      # true/false is never a number here
        return False
    if not isinstance(value, int) and (whole or not isinstance(value, float)):
        return False
    # A whole number is compared as it is: one too large for a float (1 followed by 400 zeros) is
    # simply out of range, where math.isfinite would raise. Only a float can be NaN or infinite.
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return low <= value <= high


def learning_problems(body: Any) -> List[Dict[str, str]]:
    """What is wrong with a request to change the learning thresholds, in plain words: one entry
    per field, the seven thresholds first (in their order), then anything that is not one. The
    body must be exactly the seven keys. Empty when it can be saved as it is. Pure."""
    body = body if isinstance(body, dict) else {}
    problems = []
    for key in LEARNING_KEYS:
        if key not in body or not _in_range(key, body[key]):
            whole, low, high = LEARNING_RANGES[key]
            problems.append({"field": key,
                             "message": f"Enter a {'whole number' if whole else 'number'} from {low} to {high}."})
    problems += [{"field": str(key), "message": "This is not a learning threshold."}
                 for key in sorted(map(str, body)) if key not in LEARNING_KEYS]
    return problems


def learning_of(settings: Dict[str, Any]) -> Dict[str, Any]:
    """The seven thresholds of (merged) settings, as stored: a value someone wrote straight into
    the row that is not a number is shown as it is, never replaced by a guess."""
    learning = settings.get("learning") if isinstance(settings.get("learning"), dict) else {}
    return {key: learning.get(key) for key in LEARNING_KEYS}


def save_learning(conn, values: Dict[str, Any], *, actor: str,
                  record: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Write the seven learning thresholds into the settings row and return (before, after).

    One transaction (get_conn() is AUTOCOMMIT): the row is created empty if it does not exist,
    locked, read, and written back with `learning`'s seven keys replaced. Every other key of the
    row, and anything else stored under `learning`, is written back as it was read. `record(before,
    after)` runs inside the transaction, so a change whose audit record cannot be written is not
    made. `before` is what was in force: the stored values over the ruled defaults.

    Raises ValueError when the stored value is not a JSON object: there is nothing to merge into,
    and overwriting it would lose whatever it holds. `values` are taken as already checked
    (`learning_problems`)."""
    from services.agent_policy import approvals   # lazily: approvals pulls in the decision engine

    after = {key: values[key] for key in LEARNING_KEYS}
    with approvals._tx(conn):
        with conn.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) "
                        "VALUES (%s, '{}'::jsonb, %s) ON CONFLICT (config_key) DO NOTHING", (CONFIG_KEY, actor))
            cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = %s FOR UPDATE",
                        (CONFIG_KEY,))
            stored = cur.fetchone()[0]
            if isinstance(stored, str):
                stored = json.loads(stored)
            if not isinstance(stored, dict):
                raise ValueError(f"the stored {CONFIG_KEY} is not a JSON object; it is left as it is")
            before = learning_of(merge(stored))
            kept = stored.get("learning") if isinstance(stored.get("learning"), dict) else {}
            cur.execute("UPDATE proc.bp_admin_config SET config_value = %s::jsonb, last_modified_by = %s, "
                        "last_modified_date = now() WHERE config_key = %s",
                        (json.dumps({**stored, "learning": {**kept, **after}}), actor, CONFIG_KEY))
            record(before, after)
    return before, after


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
