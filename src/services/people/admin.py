"""Admin and self-service operations on people, routing rules, notifications and preferences.

Validation is pure (it takes a PeopleStore) so it is tested without a database; the SQL is thin.
Nothing here decides who MAY call it: the routers do that (gate for admin, signed-in for own data).
bp_person.roles is a mirror used only to find an admin fallback when routing a notification. It is
NOT an authorisation source; permissions stay in the auth layer.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from services.people.events import CATALOGUE
from services.people.hierarchy import HierarchyService
from services.people.models import PeopleStore, Person
from services.people.resolver import TARGETS

TENANT = "default"          # single-tenant today; every query below is scoped by it anyway
PERSON_FIELDS = ("display_name", "email", "title", "manager_subject", "department_id", "is_department_head",
                 "roles", "approval_limit", "is_active", "delegate_subject", "delegate_until")
CHANNELS = {"email"}
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class PeopleError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _parse_dt(v: Any) -> Optional[datetime]:
    if v in (None, ""):
        return None
    try:
        d = v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        raise PeopleError(422, "delegate_until must be an ISO date-time")
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def validate_person_update(store: PeopleStore, tenant: str, subject: str, fields: Dict[str, Any],
                           now: Optional[datetime] = None) -> Dict[str, Any]:
    """Return the cleaned fields to write, or raise PeopleError. Unknown fields are refused."""
    unknown = sorted(set(fields) - set(PERSON_FIELDS))
    if unknown:
        raise PeopleError(422, f"cannot set: {', '.join(unknown)}")
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        if k in ("display_name", "email", "department_id", "title"):
            out[k] = (str(v).strip() or None) if v is not None else None
        elif k in ("is_department_head", "is_active"):
            if not isinstance(v, bool):
                raise PeopleError(422, f"{k} must be true or false")
            out[k] = v
        elif k == "roles":
            if not isinstance(v, list) or not all(isinstance(r, str) and r.strip() for r in v):
                raise PeopleError(422, "roles must be a list of names")
            out[k] = sorted({r.strip() for r in v})
        elif k == "approval_limit":
            if v is None:
                out[k] = None
            else:
                try:
                    d = Decimal(str(v))
                except InvalidOperation:
                    raise PeopleError(422, "approval_limit must be a number (-1 means unlimited)")
                if d < 0 and d != -1:
                    raise PeopleError(422, "approval_limit must be 0 or more, or -1 for unlimited")
                out[k] = d
        elif k == "manager_subject":
            out[k] = _check_manager(store, tenant, subject, v)
        elif k == "delegate_subject":
            out[k] = _check_delegate(store, tenant, subject, v)
        elif k == "delegate_until":
            out[k] = _parse_dt(v)
    if out.get("delegate_subject") and "delegate_until" in fields and out.get("delegate_until") is None:
        raise PeopleError(422, "a delegate needs an end date")
    if "delegate_until" in out and out["delegate_until"] and out["delegate_until"] <= (now or datetime.now(timezone.utc)):
        raise PeopleError(422, "delegate_until must be in the future")
    if "delegate_subject" in out and out["delegate_subject"] is None:
        out["delegate_until"] = None
    return out


def _check_manager(store: PeopleStore, tenant: str, subject: str, manager: Any) -> Optional[str]:
    if manager in (None, ""):
        return None
    manager = str(manager)
    if manager == subject:
        raise PeopleError(422, "a person cannot be their own manager")
    m = store.get(tenant, manager)
    if m is None:
        raise PeopleError(422, "that manager is not a known person; add them first")
    if any(p.subject == subject for p in HierarchyService(store).get_chain_of_command(m)):
        raise PeopleError(422, "that would make a reporting loop")
    return manager


def _check_delegate(store: PeopleStore, tenant: str, subject: str, delegate: Any) -> Optional[str]:
    if delegate in (None, ""):
        return None
    delegate = str(delegate)
    if delegate == subject:
        raise PeopleError(422, "a person cannot delegate to themselves")
    d = store.get(tenant, delegate)
    if d is None or not d.is_active:
        raise PeopleError(422, "the delegate must be an active, known person")
    return delegate


def validate_rule(event_type: str, targets: Any, notify_actor: Any) -> tuple:
    if event_type not in CATALOGUE:
        raise PeopleError(404, f"{event_type} is not a known event")
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) for t in targets):
        raise PeopleError(422, "targets must be a non-empty list")
    bad = sorted(set(targets) - TARGETS)
    if bad:
        raise PeopleError(422, f"unknown targets: {', '.join(bad)}; allowed: {', '.join(sorted(TARGETS))}")
    if not isinstance(notify_actor, bool):
        raise PeopleError(422, "notify_actor must be true or false")
    return list(dict.fromkeys(targets)), notify_actor


def validate_prefs(prefs: Any) -> Dict[str, Any]:
    if not isinstance(prefs, dict):
        raise PeopleError(422, "preferences must be an object")
    extra = sorted(set(prefs) - {"channels", "quiet_hours", "muted_events"})
    if extra:
        raise PeopleError(422, f"unknown preference: {', '.join(extra)}")
    out: Dict[str, Any] = {}
    if "channels" in prefs:
        ch = prefs["channels"]
        if not isinstance(ch, list) or set(ch) - CHANNELS:
            raise PeopleError(422, f"channels may only be: {', '.join(sorted(CHANNELS))} (in-app is always on)")
        out["channels"] = sorted(set(ch))
    if "quiet_hours" in prefs:
        q = prefs["quiet_hours"]
        out["quiet_hours"] = None            # None clears it
        if q is not None:
            if not (isinstance(q, list) and len(q) == 2 and all(isinstance(x, str) and _HHMM.match(x) for x in q)):
                raise PeopleError(422, "quiet_hours must be [start, end] as HH:MM")
            out["quiet_hours"] = q
    if "muted_events" in prefs:
        m = prefs["muted_events"]
        if not isinstance(m, list) or set(m) - set(CATALOGUE):
            raise PeopleError(422, "muted_events must be known event types")
        mandatory = sorted(e for e in m if CATALOGUE[e][1])
        if mandatory:
            raise PeopleError(422, f"these cannot be muted: {', '.join(mandatory)}")
        out["muted_events"] = sorted(set(m))
    return out


# ------------------------------------------------------------------ SQL
_P = ("subject, display_name, email, manager_subject, department_id, is_department_head, roles, "
      "approval_limit, is_active, delegate_subject, delegate_until, title, last_modified_by, last_modified_at")


def _person_dict(r) -> Dict[str, Any]:
    keys = [c.strip() for c in _P.split(",")]
    d = dict(zip(keys, r))
    d["approval_limit"] = None if d["approval_limit"] is None else float(d["approval_limit"])
    for k in ("delegate_until", "last_modified_at"):
        d[k] = d[k].isoformat() if d[k] else None
    d["roles"] = list(d["roles"] or [])
    return d


def list_people(conn, department: Optional[str] = None, include_inactive: bool = False) -> List[Dict[str, Any]]:
    where, args = ["tenant_id=%s"], [TENANT]
    if department:
        where.append("department_id=%s"); args.append(department)
    if not include_inactive:
        where.append("is_active")
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_P} FROM proc.bp_person WHERE {' AND '.join(where)} ORDER BY subject", args)
        return [_person_dict(r) for r in cur.fetchall()]


def get_person(conn, subject: str) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_P} FROM proc.bp_person WHERE tenant_id=%s AND subject=%s", (TENANT, subject))
        r = cur.fetchone()
        return _person_dict(r) if r else None


def upsert_person(conn, subject: str, clean: Dict[str, Any], by: str) -> Dict[str, Any]:
    cols = list(clean) + ["last_modified_by", "last_modified_at"]
    vals = list(clean.values()) + [by, datetime.now(timezone.utc)]
    ins_cols = ["tenant_id", "subject"] + cols
    ph = ", ".join(["%s"] * len(ins_cols))
    sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO proc.bp_person ({', '.join(ins_cols)}) VALUES ({ph}) "
                    f"ON CONFLICT (tenant_id, subject) DO UPDATE SET {sets}", [TENANT, subject] + vals)
    return get_person(conn, subject)


def list_rules(conn) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT event_type, targets, notify_actor, tenant_id, last_modified_by FROM proc.bp_routing_rule "
                    "WHERE tenant_id IN (%s,'default') ORDER BY event_type, (tenant_id='default')", (TENANT,))
        rows = cur.fetchall()
    seen, out = set(), []
    for et, targets, na, tid, by in rows:
        if et in seen:
            continue
        seen.add(et)
        out.append({"event_type": et, "targets": list(targets), "notify_actor": na, "set_by": by})
    return out


def put_rule(conn, event_type: str, targets: List[str], notify_actor: bool, by: str) -> None:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_routing_rule (tenant_id,event_type,targets,notify_actor,last_modified_by,"
                    "last_modified_at) VALUES (%s,%s,%s,%s,%s,now()) ON CONFLICT (tenant_id,event_type) DO UPDATE "
                    "SET targets=EXCLUDED.targets, notify_actor=EXCLUDED.notify_actor, "
                    "last_modified_by=EXCLUDED.last_modified_by, last_modified_at=now()",
                    (TENANT, event_type, targets, notify_actor, by))


def delete_rule(conn, event_type: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_routing_rule WHERE tenant_id=%s AND event_type=%s", (TENANT, event_type))
        return cur.rowcount


# ---- own notifications / preferences
def list_notifications(conn, subject: str, unread_only: bool, limit: int, before_id: Optional[int]) -> Dict[str, Any]:
    where, args = ["tenant_id=%s", "user_subject=%s"], [TENANT, subject]
    if unread_only:
        where.append("read_at IS NULL")
    if before_id:
        where.append("notification_id < %s"); args.append(before_id)
    with conn.cursor() as cur:
        cur.execute("SELECT notification_id, event_id, reason, detail, title, body, link, priority, read_at, created_at "
                    f"FROM proc.bp_notification WHERE {' AND '.join(where)} ORDER BY notification_id DESC LIMIT %s",
                    args + [limit])
        items = [{"id": r[0], "event_id": r[1], "reason": r[2], "why": r[3], "title": r[4], "body": r[5],
                  "link": r[6], "priority": r[7], "read": r[8] is not None,
                  "created_at": r[9].isoformat()} for r in cur.fetchall()]
    return {"items": items, "unread": unread_count(conn, subject),
            "next_before_id": items[-1]["id"] if len(items) == limit else None}


def unread_count(conn, subject: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_notification WHERE tenant_id=%s AND user_subject=%s AND read_at IS NULL",
                    (TENANT, subject))
        return int(cur.fetchone()[0])


def mark_read(conn, subject: str, notification_id: int) -> bool:
    """Only the owner's own row; someone else's id is simply 'not found'."""
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_notification SET read_at = COALESCE(read_at, now()) "
                    "WHERE notification_id=%s AND tenant_id=%s AND user_subject=%s", (notification_id, TENANT, subject))
        return cur.rowcount == 1


def mark_all_read(conn, subject: str) -> int:
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_notification SET read_at=now() WHERE tenant_id=%s AND user_subject=%s "
                    "AND read_at IS NULL", (TENANT, subject))
        return cur.rowcount


def get_prefs(conn, subject: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT pref_value FROM proc.bp_user_preference WHERE user_subject=%s AND pref_key='notifications'",
                    (subject,))
        r = cur.fetchone()
    return dict(r[0]) if r and isinstance(r[0], dict) else {}


def put_prefs(conn, subject: str, clean: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**get_prefs(conn, subject), **clean}
    for k in [k for k, v in merged.items() if v is None]:
        del merged[k]
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_user_preference (user_subject,pref_key,pref_value,updated_at) "
                    "VALUES (%s,'notifications',%s::jsonb,now()) ON CONFLICT (user_subject,pref_key) "
                    "DO UPDATE SET pref_value=EXCLUDED.pref_value, updated_at=now()", (subject, json.dumps(merged)))
    return merged
