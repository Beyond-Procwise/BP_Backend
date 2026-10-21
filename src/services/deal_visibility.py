"""Onboarding visibility setting: which deals each role may see, and which it may change.

Decision D-36: this is an administrator's configuration, chosen at onboarding, not a rule in the
code. Reading and writing are separate. The starting value is the narrow one: a person sees (and
changes) only the deals they created or approved. An administrator may widen a role to ``all``
(every deal in the company).

Stored as one ``proc.bp_policy`` row, ``DealVisibilityPolicy``, in the same way as the permission
settings (services/permission_settings): one transaction, optimistic lock, a history line per
change, an audit record, and seeding that never overwrites a choice.

  * Scopes: read is ``mine`` or ``all``; write is ``none``, ``mine`` or ``all``.
  * A role with no entry, a value that is not a known scope, or a row that cannot be read gets the
    narrow scope. An outage must never look like "everyone sees everything".
  * Admin is always ``all``; narrowing it is refused, since that could lock everyone out.
  * This module decides the SCOPE. Checking a given deal against it needs to know who created or
    approved the deal; ``allows`` takes those people as arguments.

Rows take effect when read: nothing here is cached.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from services.db import get_conn
from services.permission_settings import (InvalidRole, SettingConflict, _in_transaction, _known_roles,
                                          _load, _record)

POLICY_NAME = "DealVisibilityPolicy"
POLICY_IDENTIFIER = "deal_visibility"
POLICY_TYPE = "security"
DESCRIPTION = ("Which deals each role may see and change. Default: only the deals the person "
               "created or approved. Set a role to 'all' to let it see every deal.")

READ_SCOPES = ("mine", "all")
WRITE_SCOPES = ("none", "mine", "all")
NARROW = {"read": "mine", "write": "mine"}
ADMIN = "Admin"


class InvalidScope(ValueError):
    """The scope is not one of the allowed values for read or write."""


class AdminCannotBeNarrowed(ValueError):
    """Admin keeps every deal; narrowing it could lock everybody out."""


@dataclass(frozen=True)
class Visibility:
    role: str
    read: str
    write: str


def _default_for(role: str) -> Dict[str, str]:
    return {"read": "all", "write": "all"} if role == ADMIN else dict(NARROW)


def _details(roles: Dict[str, Dict[str, str]], history: Optional[list] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"policy_identifier": POLICY_IDENTIFIER, "rules": {"roles": roles}}
    if history:
        d["history"] = history
    return d


def _clean(entry: Any, role: str) -> Dict[str, str]:
    """One role's stored entry, with anything unknown replaced by the narrow default."""
    base = _default_for(role)
    if not isinstance(entry, dict):
        return base
    read = entry.get("read") if entry.get("read") in READ_SCOPES else base["read"]
    write = entry.get("write") if entry.get("write") in WRITE_SCOPES else base["write"]
    if role == ADMIN:
        return {"read": "all", "write": "all"}
    return {"read": read, "write": write}


def _stored(cur) -> Optional[tuple]:
    return _load(cur, [POLICY_NAME]).get(POLICY_NAME)


def current(conn=None, roles: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every defined role with its read and write scope; ``configured`` is False with no row."""
    if conn is None:
        with get_conn() as own:
            return current(own, roles)
    got = _stored(conn.cursor())
    stored = ((got[2].get("rules") or {}).get("roles") or {}) if got else {}
    out = []
    for role in _known_roles(roles):
        entry = _clean(stored.get(role), role)
        out.append({"role": role, "read": entry["read"], "write": entry["write"],
                    "configured": role in stored, "version": got[1] if got else 0})
    return out


def resolve(principal: Any, conn=None) -> Visibility:
    """The scope the signed-in person acts with. Never raises: any failure is the narrow scope."""
    role = ""
    try:
        from services import rbac
        role = rbac.effective_role(principal)
        if role == ADMIN:
            return Visibility(role, "all", "all")
        if conn is None:
            with get_conn() as own:
                got = _stored(own.cursor())
        else:
            got = _stored(conn.cursor())
        stored = ((got[2].get("rules") or {}).get("roles") or {}) if got else {}
        entry = _clean(stored.get(role), role)
        return Visibility(role, entry["read"], entry["write"])
    except Exception:  # noqa: BLE001
        return Visibility(role, NARROW["read"], NARROW["write"])


def allows(scope: str, subject: Optional[str], created_by: Iterable[str] = (),
           approved_by: Iterable[str] = ()) -> bool:
    """May a person with ``scope`` act on a deal that these people created / approved?

    ``all`` allows. ``mine`` allows only a named person who created or approved the deal. Anything
    else, and anyone unnamed, is refused.
    """
    if scope == "all":
        return True
    if scope != "mine" or not subject:
        return False
    return subject in set(created_by) or subject in set(approved_by)


def seed_defaults(by: str, conn=None, roles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create the row if there is none (every role at its default). Never overwrites a choice."""
    if conn is None:
        with get_conn() as own:
            return seed_defaults(by, own, roles)
    known = list(_known_roles(roles))

    def work(cur):
        if _load(cur, [POLICY_NAME], for_update=True).get(POLICY_NAME):
            return {"created": False}
        table = {r: _default_for(r) for r in known}
        cur.execute(
            "INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, "
            "policy_linked_agents, policy_status, version, created_by, created_date) "
            "VALUES (%s, %s, %s, %s::jsonb, '', 1, 1, %s, now())",
            (POLICY_NAME, POLICY_TYPE, DESCRIPTION, json.dumps(_details(table)), by))
        _record(conn, phase="configure", action_type="deal_visibility_seeded",
                summary=f"deal visibility defaults written by {by}",
                details={"changed_by": by, "roles": known})
        return {"created": True}
    return _in_transaction(conn, work)


def set_scope(role: str, by: str, read: Optional[str] = None, write: Optional[str] = None,
              conn=None, roles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Change one role's read and/or write scope. A save that changes nothing writes nothing."""
    if conn is None:
        with get_conn() as own:
            return set_scope(role, by, read, write, own, roles)
    known = _known_roles(roles)
    if role not in known:
        raise InvalidRole(f"{role!r} is not a defined role")
    if read is None and write is None:
        raise InvalidScope("give a read scope, a write scope, or both")
    if read is not None and read not in READ_SCOPES:
        raise InvalidScope(f"read must be one of {', '.join(READ_SCOPES)}")
    if write is not None and write not in WRITE_SCOPES:
        raise InvalidScope(f"write must be one of {', '.join(WRITE_SCOPES)}")
    if role == ADMIN and (read not in (None, "all") or write not in (None, "all")):
        raise AdminCannotBeNarrowed("Admin always sees and changes every deal")

    def work(cur):
        have = _load(cur, [POLICY_NAME], for_update=True).get(POLICY_NAME)
        details = have[2] if have else _details({r: _default_for(r) for r in known})
        version = have[1] if have else 0
        table = {r: _clean(e, r) for r, e in ((details.get("rules") or {}).get("roles") or {}).items()}
        for r in known:
            table.setdefault(r, _default_for(r))
        before = dict(table[role])
        after = {"read": read or before["read"], "write": write or before["write"]}
        if have and before == after and role in (details.get("rules") or {}).get("roles", {}):
            return {"role": role, **after, "changed": False, "version": version}
        now = datetime.now(timezone.utc).isoformat()
        history = list(details.get("history") or []) + [
            {"version": version + 1, "changed_by": by, "changed_at": now, "field": f"{role}.{f}",
             "before": before[f], "after": after[f]} for f in ("read", "write") if before[f] != after[f]]
        table[role] = after
        merged = _details(table, history)
        if have:
            cur.execute(
                "UPDATE proc.bp_policy SET policy_details = %s::jsonb, version = version + 1, "
                "last_modified_by = %s, last_modified_date = now() WHERE policy_id = %s AND version = %s",
                (json.dumps(merged), by, have[0], version))
            if cur.rowcount != 1:
                raise SettingConflict("deal visibility was changed by somebody else; reload and try again")
        else:
            cur.execute(
                "INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, "
                "policy_linked_agents, policy_status, version, created_by, created_date) "
                "VALUES (%s, %s, %s, %s::jsonb, '', 1, 1, %s, now())",
                (POLICY_NAME, POLICY_TYPE, DESCRIPTION, json.dumps(merged), by))
        _record(conn, phase="configure", action_type="deal_visibility_changed",
                summary=f"deal visibility for {role}: read {before['read']}->{after['read']}, "
                        f"write {before['write']}->{after['write']} by {by}",
                details={"changed_by": by, "role": role, "before": before, "after": after,
                         "version": version + 1})
        return {"role": role, **after, "changed": True, "version": version + 1}
    return _in_transaction(conn, work)
