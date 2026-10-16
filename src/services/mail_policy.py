"""The per-customer email policy: read strictly, validate, save with history.

The policy lives in one ``proc.bp_policy`` row (``policy_name='email_policy'``).
Nothing here carries a default: a missing or invalid value raises
``PolicyUnavailable`` so an outage can never look like "nothing is restricted".
Every applied save writes one history entry per changed field and an audit row,
in the same transaction -- if either cannot be written, the save does not apply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from services.agent_actions import record_action_or_fail
from services.db import get_conn


class PolicyUnavailable(RuntimeError):
    """The email policy row, or one of its values, is missing or invalid."""


class PolicyConflict(RuntimeError):
    """The policy was changed by somebody else since it was read."""


_RE_GROUP = re.compile(r"[A-Za-z0-9 ._-]{1,64}")
_RE_DOMAIN = re.compile(r"(?=.{1,253}\Z)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")


def _choice(*allowed):
    def v(x):
        if x not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)}")
        return x
    return v


def _domains(x):
    if not isinstance(x, list):
        raise ValueError("must be a list of domains")
    out = []
    for d in x:
        d = str(d).strip().lower().lstrip("@")
        if not _RE_DOMAIN.fullmatch(d):
            raise ValueError(f"{d!r} is not a domain")
        if d not in out:
            out.append(d)
    return tuple(out)


def _poll(x):
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError("must be a whole number of seconds")
    if not 60 <= x <= 3600:
        raise ValueError("must be between 60 and 3600 seconds")
    return x


def _group(x):
    if not isinstance(x, str):
        raise ValueError("must be text")
    if not _RE_GROUP.fullmatch(x):
        raise ValueError("letters, numbers, spaces, . _ - only (max 64)")
    return x


FIELDS: Dict[str, Callable[[Any], Any]] = {
    "internal_send": _choice("off", "with_user_approval"),
    "company_domains": _domains,
    "external_recipients": _choice("known_suppliers_only", "any"),
    "read_scope": _choice("replies_and_suppliers", "whole_inbox"),
    "on_disconnect": _choice("hold", "procwise_address"),
    "access_level": _choice("read_send", "read_send_change"),
    "notices_from_mailbox": _choice("off", "with_user_approval"),
    "poll_seconds": _poll,
    "restriction_group": _group,
}

_NEEDS_DOMAINS = "Add at least one company domain before allowing emails to colleagues"


@dataclass(frozen=True)
class EmailPolicy:
    version: int
    internal_send: str
    company_domains: Tuple[str, ...]
    external_recipients: str
    read_scope: str
    on_disconnect: str
    access_level: str
    notices_from_mailbox: str
    poll_seconds: int
    restriction_group: str
    history: Tuple[dict, ...]


_SELECT = """SELECT policy_id, version, policy_details FROM proc.bp_policy
              WHERE policy_name = 'email_policy' AND policy_status = 1"""


def _fetch(cur, for_update: bool = False):
    cur.execute(_SELECT + (" FOR UPDATE" if for_update else ""))
    row = cur.fetchone()
    if row is None:
        raise PolicyUnavailable("email policy: the policy row is missing")
    policy_id, version, details = row[0], row[1], row[2]
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError as exc:
            raise PolicyUnavailable(f"email policy: policy_details is not valid JSON: {exc}") from exc
    if not isinstance(details, dict):
        raise PolicyUnavailable("email policy: policy_details is not an object")
    return policy_id, int(version or 0), details


def _history(details: dict) -> list:
    history = details.get("history", [])
    if history is None:
        history = []
    if not isinstance(history, list) or not all(isinstance(h, dict) for h in history):
        raise PolicyUnavailable("email policy: history: must be a list of entries")
    return history


def _build(version: int, details: dict) -> EmailPolicy:
    rules = details.get("rules")
    if not isinstance(rules, dict):
        raise PolicyUnavailable("email policy: rules: missing")
    values: Dict[str, Any] = {}
    for key, check in FIELDS.items():
        try:
            values[key] = check(rules[key])
        except KeyError:
            raise PolicyUnavailable(f"email policy: {key}: missing") from None
        except (ValueError, TypeError) as exc:
            raise PolicyUnavailable(f"email policy: {key}: {exc}") from exc
    history = _history(details)
    return EmailPolicy(version=version, history=tuple(history), **values)


def load(conn=None) -> EmailPolicy:
    """Read the policy; raises ``PolicyUnavailable`` if anything is missing or invalid."""
    if conn is not None:
        _, version, details = _fetch(conn.cursor())
        return _build(version, details)
    with get_conn() as own:
        _, version, details = _fetch(own.cursor())
        return _build(version, details)


def _json_value(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def save(changes: Dict[str, Any], *, by: str, expected_version: int, conn=None) -> EmailPolicy:
    """Validate and store ``changes``; one history entry per changed field."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("no settings were given to change")
    cleaned: Dict[str, Any] = {}
    for key, value in changes.items():
        if key not in FIELDS:
            raise ValueError(f"unknown setting: {key}")
        try:
            cleaned[key] = _json_value(FIELDS[key](value))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{key}: {exc}") from exc

    if conn is not None:
        return _save(conn, cleaned, by, expected_version)
    with get_conn() as own:
        return _save(own, cleaned, by, expected_version)


def _save(conn, cleaned: Dict[str, Any], by: str, expected_version: int) -> EmailPolicy:
    # get_conn() connections are autocommit; the lock, the update, the history and
    # the audit row must stand or fall together, so run them as one transaction.
    previous = getattr(conn, "autocommit", True)
    conn.autocommit = False
    try:
        try:
            cur = conn.cursor()
            policy_id, version, details = _fetch(cur, for_update=True)
            if version != int(expected_version):
                raise PolicyConflict("the email settings were changed by somebody else; reload and try again")
            history = _history(details)
            if not isinstance(details.get("rules"), dict):
                raise PolicyUnavailable("email policy: rules: missing")
            rules = dict(details["rules"])
            diff = []
            for key, after in cleaned.items():
                before = rules.get(key)
                if before != after:
                    diff.append({"field": key, "before": before, "after": after})
                rules[key] = after
            merged = {**details, "rules": rules}
            candidate = _build(version + 1, merged)  # strict: the whole result must hold together
            if ((candidate.internal_send != "off" or candidate.notices_from_mailbox != "off")
                    and not candidate.company_domains):
                raise ValueError(_NEEDS_DOMAINS)
            if not diff:
                conn.rollback()
                return _build(version, details)
            now = datetime.now(timezone.utc).isoformat()
            history = history + [
                {"version": version + 1, "changed_by": by, "changed_at": now, **d} for d in diff]
            merged["history"] = history
            # The version is in the WHERE as well as the check above: a stale save
            # must change nothing rather than overwrite the one that beat it.
            cur.execute(
                """UPDATE proc.bp_policy
                      SET policy_details = %s::jsonb, version = version + 1,
                          last_modified_by = %s, last_modified_date = now()
                    WHERE policy_id = %s AND version = %s""",
                (json.dumps(merged), by, policy_id, version))
            if cur.rowcount != 1:
                raise PolicyConflict("the email settings were changed by somebody else; reload and try again")
            record_action_or_fail(
                phase="configure", action_type="email_policy_saved", conn=conn,
                summary=f"email policy changed by {by}: " + ", ".join(d["field"] for d in diff),
                details={"changed_by": by, "version": version + 1, "changes": diff})
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        conn.autocommit = previous
    return _build(version + 1, merged)
