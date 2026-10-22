"""Handing a piece of work to a named person (D-73, second half: findings).

An approval is handed on through the approvals service (delegation.py); everything else that a
person works — a finding today — is handed on here. One active assignment per item: handing it
to someone else releases the old row and writes a new one, so every hand stays on record. The
receiver is told by a task.assigned notification that says what the item is and carries a link to
it; handing an item to yourself is claiming it and tells nobody. Who may hand a finding on is the
endpoint gate's call (finding.triage, a write); the receiver must be an active person in the chart.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from services.people.admin import PeopleError, TENANT
from services.people.events import DomainEvent
from services.people.db import tx

EVENT_TYPE = "task.assigned"
MAX_NOTE = 2000
ENTITY_TYPES = ("finding",)

_COLS = ("assignment_id, entity_type, entity_id, assignee_subject, assigned_by, assigned_at, note, "
         "released_at, released_by")


def clean_note(note: Any) -> Optional[str]:
    note = (note or "").strip() if isinstance(note, str) else None
    if note and len(note) > MAX_NOTE:
        raise PeopleError(422, f"a note can be at most {MAX_NOTE} characters")
    return note or None


def check_entity_type(entity_type: str) -> str:
    if entity_type not in ENTITY_TYPES:
        raise PeopleError(422, f"'{entity_type}' is not something that can be handed on; today: {', '.join(ENTITY_TYPES)}")
    return entity_type


def what_finding(f: Dict[str, Any]) -> str:
    """What the finding is, in words and without its values: 'duplicate invoice on invoice X.pdf'."""
    issue = str(f.get("issue_type") or "finding").replace("_", " ")
    doc = str(f.get("doc_type") or "document")
    name = os.path.basename(str(f.get("source_file") or "")) or "a document"
    return f"{issue} on {doc} {name}"


def about_finding(f: Dict[str, Any]) -> str:
    """The context line under the notification's title."""
    return f"About finding {f['discrepancy_id']}: {what_finding(f)}"


def link_for(entity_type: str, entity_id: str) -> str:
    return f"/actions/queue?{entity_type}={entity_id}"


def _finding(cur, entity_id: str) -> Dict[str, Any]:
    try:
        fid = int(str(entity_id).strip())
    except ValueError:
        raise PeopleError(404, "no such finding")
    cur.execute("SELECT discrepancy_id, issue_type, doc_type, source_file, severity, status "
                "FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (fid,))
    r = cur.fetchone()
    if r is None:
        raise PeopleError(404, "no such finding")
    return dict(zip(("discrepancy_id", "issue_type", "doc_type", "source_file", "severity", "status"), r))


def _person(cur, subject: str) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT subject, display_name, is_active FROM proc.bp_person WHERE tenant_id=%s AND subject=%s",
                (TENANT, subject))
    r = cur.fetchone()
    return {"subject": r[0], "name": r[1] or r[0], "active": bool(r[2])} if r else None


def _row(r) -> Dict[str, Any]:
    d = dict(zip([c.strip() for c in _COLS.split(",")], r))
    for k in ("assigned_at", "released_at"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    return d


def assign(conn, me: str, entity_type: str, entity_id: str, to_subject: str, note: Any = None,
           now: Optional[datetime] = None) -> Dict[str, Any]:
    """Hand the item to `to_subject` (yourself = claim it). Returns the active assignment."""
    check_entity_type(entity_type)
    to_subject = str(to_subject or "").strip()
    if not to_subject:
        raise PeopleError(422, "say who should work this")
    note = clean_note(note)
    now = now or datetime.now(timezone.utc)
    from services.people.wiring import emit_safe
    with tx(conn), conn.cursor() as cur:
        item = _finding(cur, entity_id)
        who = _person(cur, to_subject)
        if who is None or not who["active"]:
            raise PeopleError(422, "that person is not an active, known person")
        by = _person(cur, me)
        by_name = by["name"] if by else me
        cur.execute("UPDATE proc.bp_work_assignment SET released_at=%s, released_by=%s "
                    "WHERE tenant_id=%s AND entity_type=%s AND entity_id=%s AND released_at IS NULL",
                    (now, me, TENANT, entity_type, str(entity_id)))
        cur.execute(f"INSERT INTO proc.bp_work_assignment (tenant_id, entity_type, entity_id, assignee_subject, "
                    f"assigned_by, assigned_at, note) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING {_COLS}",
                    (TENANT, entity_type, str(entity_id), who["subject"], me, now, note))
        row = _row(cur.fetchone())
        if who["subject"] != me:
            about = about_finding(item)
            emit_safe(cur, DomainEvent(
                type=EVENT_TYPE, actor_id=me, entity_type=entity_type, entity_id=str(entity_id),
                summary=f"{by_name} handed you a finding: {what_finding(item)}",
                event_id=f"{EVENT_TYPE}:{entity_type}:{entity_id}:{row['assignment_id']}",
                data={"explicit": [who["subject"]], "link": link_for(entity_type, str(entity_id)),
                      "body": f"{note}\n{about}" if note else about}))
    row["assignee_name"] = who["name"]
    row["by_name"] = by_name
    return row


def release(conn, me: str, entity_type: str, entity_id: str, now: Optional[datetime] = None) -> bool:
    """Nobody works it any more. True when there was an active assignment."""
    check_entity_type(entity_type)
    now = now or datetime.now(timezone.utc)
    with tx(conn), conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_work_assignment SET released_at=%s, released_by=%s "
                    "WHERE tenant_id=%s AND entity_type=%s AND entity_id=%s AND released_at IS NULL",
                    (now, me, TENANT, entity_type, str(entity_id)))
        return cur.rowcount > 0


def active(conn, entity_type: str, ids: Optional[Iterable[str]] = None, limit: int = 2000) -> List[Dict[str, Any]]:
    """The active assignments of a kind, optionally only for the given ids, with names."""
    check_entity_type(entity_type)
    wanted = [str(i).strip() for i in (ids or []) if str(i).strip()]
    with conn.cursor() as cur:
        sql = (f"SELECT {', '.join('a.' + c.strip() for c in _COLS.split(','))}, p.display_name, q.display_name "
               "FROM proc.bp_work_assignment a "
               "LEFT JOIN proc.bp_person p ON p.tenant_id=a.tenant_id AND p.subject=a.assignee_subject "
               "LEFT JOIN proc.bp_person q ON q.tenant_id=a.tenant_id AND q.subject=a.assigned_by "
               "WHERE a.tenant_id=%s AND a.entity_type=%s AND a.released_at IS NULL")
        params: List[Any] = [TENANT, entity_type]
        if wanted:
            sql += " AND a.entity_id = ANY(%s)"
            params.append(wanted)
        sql += " ORDER BY a.assigned_at DESC LIMIT %s"
        params.append(int(limit))
        cur.execute(sql, params)
        out = []
        for r in cur.fetchall():
            d = _row(r[:-2])
            d["assignee_name"] = r[-2] or d["assignee_subject"]
            d["by_name"] = r[-1] or d["assigned_by"]
            out.append(d)
        return out


def mine(conn, me: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Everything currently with the signed-in person, newest first."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_COLS} FROM proc.bp_work_assignment WHERE tenant_id=%s AND assignee_subject=%s "
                    "AND released_at IS NULL ORDER BY assigned_at DESC LIMIT %s", (TENANT, me, int(limit)))
        return [_row(r) for r in cur.fetchall()]
