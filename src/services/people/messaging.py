"""Conversations, participants and messages.

Access rule, one place: a conversation is visible only to its current participants. Direct and
group conversations are joined by invitation; an entity thread (a discussion on a PO, contract,
supplier...) is invite-only too, because this backend has no per-item visibility rule that could
decide who may read a thread about it. Mentioning someone in an entity thread invites them, which
is an explicit act by a participant; in direct/group conversations a mention of an outsider is refused.

Notifications carry no message text (title says who wrote, nothing more): the text lives here, where
participation is checked on every read.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from services.people.admin import TENANT, PeopleError
from services.people.db import tx
from services.people.events import DomainEvent
from services.people.models import Person

MAX_BODY = 10_000
MAX_MENTIONS = 20
MAX_ATTACHMENTS = 10
MAX_GROUP = 100
ENTITY_TYPES = {"po", "contract", "supplier", "rfq", "deal", "requisition", "invoice", "approval"}
_MENTION = re.compile(r"@\[([^\]]{1,80})\]\(([^)\s]{1,128})\)")


# ------------------------------------------------------------------ pure helpers
def direct_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def clean_body(body: Any) -> str:
    if not isinstance(body, str) or not body.strip():
        raise PeopleError(422, "a message cannot be empty")
    if len(body) > MAX_BODY:
        raise PeopleError(422, f"a message can be at most {MAX_BODY} characters")
    return body.strip()


def parse_mentions(body: str, explicit: Optional[Sequence[str]] = None) -> List[str]:
    """Subjects named by @[Name](subject) tokens in the text plus any sent explicitly, in order."""
    found = [m.group(2) for m in _MENTION.finditer(body)] + [str(s) for s in (explicit or [])]
    out = list(dict.fromkeys(found))
    if len(out) > MAX_MENTIONS:
        raise PeopleError(422, f"at most {MAX_MENTIONS} mentions in one message")
    return out


def clean_attachments(items: Any) -> List[Dict[str, Any]]:
    """Metadata only: a name and, optionally, a reference to a file already stored elsewhere."""
    if items in (None, []):
        return []
    if not isinstance(items, list) or len(items) > MAX_ATTACHMENTS:
        raise PeopleError(422, f"at most {MAX_ATTACHMENTS} attachments")
    out = []
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("name"), str) or not it["name"].strip():
            raise PeopleError(422, "each attachment needs a name")
        out.append({k: it[k] for k in ("name", "ref", "content_type", "size") if k in it and it[k] is not None})
    return out


# ------------------------------------------------------------------ lookups
def _person(cur, subject: str) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT subject, display_name, is_active FROM proc.bp_person WHERE tenant_id=%s AND subject=%s",
                (TENANT, subject))
    r = cur.fetchone()
    return {"subject": r[0], "name": r[1] or r[0], "active": bool(r[2])} if r else None


def _active_person(cur, subject: str, what: str = "that person") -> Dict[str, Any]:
    p = _person(cur, subject)
    if p is None or not p["active"]:
        raise PeopleError(422, f"{what} is not an active, known person")
    return p


def _conversation(cur, conversation_id: int) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT conversation_id, type, title, linked_entity_type, linked_entity_id, created_by "
                "FROM proc.bp_conversation WHERE tenant_id=%s AND conversation_id=%s", (TENANT, conversation_id))
    r = cur.fetchone()
    return None if not r else dict(id=r[0], type=r[1], title=r[2], entity_type=r[3], entity_id=r[4], created_by=r[5])


def is_participant(cur, conversation_id: int, subject: str) -> bool:
    cur.execute("SELECT 1 FROM proc.bp_conversation_participant p JOIN proc.bp_conversation c USING (conversation_id) "
                "WHERE p.conversation_id=%s AND p.user_subject=%s AND p.left_at IS NULL AND c.tenant_id=%s",
                (conversation_id, subject, TENANT))
    return cur.fetchone() is not None


def _member_or_404(cur, conversation_id: int, me: str) -> Dict[str, Any]:
    """Not a participant looks exactly like 'no such conversation'."""
    conv = _conversation(cur, conversation_id)
    if conv is None or not is_participant(cur, conversation_id, me):
        raise PeopleError(404, "no such conversation")
    return conv


def participants(cur, conversation_id: int) -> List[Dict[str, Any]]:
    cur.execute("SELECT p.user_subject, COALESCE(pe.display_name, p.user_subject), p.role, p.muted "
                "FROM proc.bp_conversation_participant p LEFT JOIN proc.bp_person pe "
                "ON pe.tenant_id=%s AND pe.subject=p.user_subject "
                "WHERE p.conversation_id=%s AND p.left_at IS NULL ORDER BY p.joined_at", (TENANT, conversation_id))
    return [{"subject": r[0], "name": r[1], "role": r[2], "muted": r[3]} for r in cur.fetchall()]


def _add(cur, conversation_id: int, subject: str, role: str = "member") -> None:
    cur.execute("INSERT INTO proc.bp_conversation_participant (conversation_id, user_subject, role) VALUES (%s,%s,%s) "
                "ON CONFLICT (conversation_id, user_subject) DO UPDATE SET left_at=NULL",
                (conversation_id, subject, role))


def _new_conversation(cur, ctype: str, created_by: str, **cols) -> int:
    cur.execute("INSERT INTO proc.bp_conversation (tenant_id,type,title,direct_key,linked_entity_type,"
                "linked_entity_id,created_by) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING conversation_id",
                (TENANT, ctype, cols.get("title"), cols.get("direct_key"), cols.get("entity_type"),
                 cols.get("entity_id"), created_by))
    return cur.fetchone()[0]


# ------------------------------------------------------------------ create
def open_direct(conn, me: str, other: str) -> Dict[str, Any]:
    if other == me:
        raise PeopleError(422, "you cannot message yourself")
    with tx(conn), conn.cursor() as cur:
        _active_person(cur, me, "you")
        _active_person(cur, other)
        key = direct_key(me, other)
        cur.execute("SELECT conversation_id FROM proc.bp_conversation WHERE tenant_id=%s AND type='direct' "
                    "AND direct_key=%s", (TENANT, key))
        r = cur.fetchone()
        if r:
            cid = r[0]
        else:
            cid = _new_conversation(cur, "direct", me, direct_key=key)
        _add(cur, cid, me, "owner"); _add(cur, cid, other)       # re-opens if either had left
        return {"id": cid, "participants": participants(cur, cid)}


def open_group(conn, me: str, title: str, members: Sequence[str]) -> Dict[str, Any]:
    title = (title or "").strip()
    if not title:
        raise PeopleError(422, "a group needs a title")
    people = list(dict.fromkeys(m for m in members if m != me))
    if not people or len(people) + 1 > MAX_GROUP:
        raise PeopleError(422, f"a group needs 1 to {MAX_GROUP - 1} other people")
    with tx(conn), conn.cursor() as cur:
        _active_person(cur, me, "you")
        for m in people:
            _active_person(cur, m, f"'{m}'")
        cid = _new_conversation(cur, "group", me, title=title[:200])
        _add(cur, cid, me, "owner")
        for m in people:
            _add(cur, cid, m)
        return {"id": cid, "participants": participants(cur, cid)}


def open_entity_thread(conn, me: str, entity_type: str, entity_id: str, title: Optional[str]) -> Dict[str, Any]:
    entity_type = (entity_type or "").strip().lower()
    entity_id = (entity_id or "").strip()
    if entity_type not in ENTITY_TYPES:
        raise PeopleError(422, f"threads can be started on: {', '.join(sorted(ENTITY_TYPES))}")
    if not entity_id or len(entity_id) > 128:
        raise PeopleError(422, "entity_id is required")
    with tx(conn), conn.cursor() as cur:
        _active_person(cur, me, "you")
        cur.execute("SELECT conversation_id FROM proc.bp_conversation WHERE tenant_id=%s AND type='entity_thread' "
                    "AND linked_entity_type=%s AND linked_entity_id=%s", (TENANT, entity_type, entity_id))
        r = cur.fetchone()
        if r:
            if not is_participant(cur, r[0], me):
                raise PeopleError(409, "a discussion already exists on this item; ask someone in it to add you")
            return {"id": r[0], "created": False, "participants": participants(cur, r[0])}
        cid = _new_conversation(cur, "entity_thread", me, title=(title or "").strip()[:200] or None,
                                entity_type=entity_type, entity_id=entity_id)
        _add(cur, cid, me, "owner")
        return {"id": cid, "created": True, "participants": participants(cur, cid)}


def add_participant(conn, me: str, conversation_id: int, subject: str) -> Dict[str, Any]:
    with tx(conn), conn.cursor() as cur:
        conv = _member_or_404(cur, conversation_id, me)
        if conv["type"] == "direct":
            raise PeopleError(422, "people cannot be added to a direct conversation; start a group instead")
        _active_person(cur, subject)
        cur.execute("SELECT count(*) FROM proc.bp_conversation_participant WHERE conversation_id=%s AND left_at IS NULL",
                    (conversation_id,))
        if cur.fetchone()[0] >= MAX_GROUP:
            raise PeopleError(422, f"a conversation holds at most {MAX_GROUP} people")
        _add(cur, conversation_id, subject)
        return {"id": conversation_id, "participants": participants(cur, conversation_id)}


def leave(conn, me: str, conversation_id: int) -> None:
    with tx(conn), conn.cursor() as cur:
        _member_or_404(cur, conversation_id, me)
        cur.execute("UPDATE proc.bp_conversation_participant SET left_at=now() WHERE conversation_id=%s AND user_subject=%s",
                    (conversation_id, me))


def set_muted(conn, me: str, conversation_id: int, muted: bool) -> None:
    with tx(conn), conn.cursor() as cur:
        _member_or_404(cur, conversation_id, me)
        cur.execute("UPDATE proc.bp_conversation_participant SET muted=%s WHERE conversation_id=%s AND user_subject=%s",
                    (muted, conversation_id, me))


# ------------------------------------------------------------------ messages
def send(conn, me: str, conversation_id: int, body: Any, mentions: Optional[Sequence[str]] = None,
         reply_to_id: Optional[int] = None, attachments: Any = None) -> Dict[str, Any]:
    """Store the message and queue its notifications in ONE transaction. Returns the message plus
    the list of people to signal live (after commit)."""
    text = clean_body(body)
    atts = clean_attachments(attachments)
    wanted = [s for s in parse_mentions(text, mentions) if s != me]
    import json
    with tx(conn), conn.cursor() as cur:
        conv = _member_or_404(cur, conversation_id, me)
        sender = _active_person(cur, me, "you")
        if reply_to_id is not None:
            cur.execute("SELECT 1 FROM proc.bp_message WHERE message_id=%s AND conversation_id=%s",
                        (reply_to_id, conversation_id))
            if cur.fetchone() is None:
                raise PeopleError(422, "the message you are replying to is not in this conversation")
        mentioned: List[str] = []
        for s in wanted:
            p = _person(cur, s)
            if p is None or not p["active"]:
                raise PeopleError(422, f"'{s}' is not an active, known person")
            if not is_participant(cur, conversation_id, s):
                if conv["type"] != "entity_thread":
                    raise PeopleError(422, f"{p['name']} is not in this conversation; add them first")
                _add(cur, conversation_id, s)          # a participant's mention invites them to the thread
            mentioned.append(s)
        cur.execute("INSERT INTO proc.bp_message (conversation_id, sender_subject, body, mentions, attachments, reply_to_id) "
                    "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING message_id, created_at",
                    (conversation_id, me, text, mentioned, json.dumps(atts), reply_to_id))
        mid, created = cur.fetchone()
        cur.execute("UPDATE proc.bp_conversation SET last_message_at=%s WHERE conversation_id=%s", (created, conversation_id))
        cur.execute("UPDATE proc.bp_conversation_participant SET last_read_message_id=%s "
                    "WHERE conversation_id=%s AND user_subject=%s", (mid, conversation_id, me))   # your own message is read
        cur.execute("SELECT user_subject, muted FROM proc.bp_conversation_participant "
                    "WHERE conversation_id=%s AND left_at IS NULL AND user_subject<>%s", (conversation_id, me))
        others = cur.fetchall()
        everyone = [s for s, _ in others]
        # a mute silences ordinary messages, never a direct mention
        plain = [s for s, muted in others if not muted and s not in mentioned]
        _log_activity(cur, conv, me, mid, conversation_id, mentioned, [x for x in everyone if x != me])
        body_text = notification_body(conv, text)
        _announce(cur, "message.received", mid, conversation_id, me, f"New message from {sender['name']}", plain, body_text)
        _announce(cur, "comment.mentioned", mid, conversation_id, me, f"{sender['name']} mentioned you", mentioned, body_text)
    return {"id": mid, "conversation_id": conversation_id, "sender": me, "body": text, "mentions": mentioned,
            "attachments": atts, "reply_to_id": reply_to_id, "created_at": created.isoformat(),
            "signal": everyone}


def _log_activity(cur, conv, me, mid, conversation_id, mentioned, others) -> None:
    """Behaviour goes to the activity store (ids and verbs, never text); it never becomes a graph edge directly."""
    from services.people.graph import activity
    ent = dict(entity_type=conv["entity_type"], entity_id=conv["entity_id"])
    direct = others[0] if conv["type"] == "direct" and others else None
    activity.record(cur, "message.sent", me, target_user_id=direct, conversation_id=conversation_id, message_id=mid,
                    conversation_type=conv["type"], **ent)
    for m in mentioned:
        activity.record(cur, "mention", me, target_user_id=m, conversation_id=conversation_id, message_id=mid, **ent)


PREVIEW_CHARS = 140


def notification_body(conv: Dict[str, Any], text: str) -> str:
    """What the recipient's notification says under the title: the start of the message, then
    what it is about. A direct message needs no context (the title names the sender); a group
    names its title; a thread on an item names the item, so the bell says what the question
    relates to before the person opens it."""
    preview = _MENTION.sub(lambda m: "@" + m.group(1), " ".join(text.split()))
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS - 1].rstrip() + "\u2026"
    kind = conv.get("type")
    if kind == "entity_thread":
        what = f"{ENTITY_LABELS.get(conv.get('entity_type'), conv.get('entity_type') or 'item')} {conv.get('entity_id')}"
        about = f"About {what}" + (f": {conv['title']}" if conv.get("title") else "")
    elif kind == "group" and conv.get("title"):
        about = f"In {conv['title']}"
    else:
        about = ""
    return f"{preview}\n{about}" if about else preview


ENTITY_LABELS = {"po": "purchase order", "contract": "contract", "supplier": "supplier", "rfq": "RFQ",
                 "deal": "deal", "requisition": "requisition", "invoice": "invoice", "approval": "approval"}


def _announce(cur, event_type: str, message_id: int, conversation_id: int, actor: str, summary: str,
              recipients: Sequence[str], body: str = "") -> None:
    if not recipients:
        return
    from services.people.wiring import emit_safe
    key = "explicit" if event_type == "message.received" else "mentioned"
    emit_safe(cur, DomainEvent(
        type=event_type, actor_id=actor, entity_type="conversation", entity_id=str(conversation_id),
        summary=summary, event_id=f"{event_type}:{message_id}",
        data={key: list(recipients), "link": f"/messages/{conversation_id}", "message_id": message_id,
              "body": body or summary}))


def _message_dict(r) -> Dict[str, Any]:
    deleted = r[7] is not None
    return {"id": r[0], "sender": r[1], "sender_name": r[8], "body": "" if deleted else r[2],
            "mentions": [] if deleted else list(r[3] or []), "attachments": [] if deleted else r[4],
            "reply_to_id": r[5], "created_at": r[6].isoformat(), "deleted": deleted,
            "edited_at": r[9].isoformat() if r[9] else None}


def list_messages(conn, me: str, conversation_id: int, before_id: Optional[int], limit: int) -> Dict[str, Any]:
    with conn.cursor() as cur:
        _member_or_404(cur, conversation_id, me)
        cur.execute("SELECT m.message_id, m.sender_subject, m.body, m.mentions, m.attachments, m.reply_to_id, m.created_at, "
                    "m.deleted_at, COALESCE(pe.display_name, m.sender_subject), m.edited_at FROM proc.bp_message m "
                    "LEFT JOIN proc.bp_person pe ON pe.tenant_id=%s AND pe.subject=m.sender_subject "
                    "WHERE m.conversation_id=%s AND (%s::bigint IS NULL OR m.message_id < %s) "
                    "ORDER BY m.message_id DESC LIMIT %s", (TENANT, conversation_id, before_id, before_id, limit))
        msgs = [_message_dict(r) for r in cur.fetchall()]
    return {"messages": msgs, "next_before_id": msgs[-1]["id"] if len(msgs) == limit else None}


def edit(conn, me: str, message_id: int, body: Any) -> Dict[str, Any]:
    text = clean_body(body)
    with tx(conn), conn.cursor() as cur:
        cur.execute("SELECT m.conversation_id, m.sender_subject, m.deleted_at FROM proc.bp_message m WHERE m.message_id=%s",
                    (message_id,))
        r = cur.fetchone()
        if r is None or not is_participant(cur, r[0], me):
            raise PeopleError(404, "no such message")
        if r[1] != me:
            raise PeopleError(403, "you can only edit your own messages")
        if r[2] is not None:
            raise PeopleError(409, "that message was deleted")
        cur.execute("UPDATE proc.bp_message SET body=%s, edited_at=now() WHERE message_id=%s", (text, message_id))
        return {"id": message_id, "conversation_id": r[0], "edited": True}


def delete(conn, me: str, message_id: int) -> Dict[str, Any]:
    """Soft delete: the row stays (replies point at it) but the text is never served again."""
    with tx(conn), conn.cursor() as cur:
        cur.execute("SELECT conversation_id, sender_subject FROM proc.bp_message WHERE message_id=%s", (message_id,))
        r = cur.fetchone()
        if r is None or not is_participant(cur, r[0], me):
            raise PeopleError(404, "no such message")
        if r[1] != me:
            raise PeopleError(403, "you can only delete your own messages")
        cur.execute("UPDATE proc.bp_message SET deleted_at=COALESCE(deleted_at, now()), body='' WHERE message_id=%s", (message_id,))
        return {"id": message_id, "conversation_id": r[0], "deleted": True}


def mark_read(conn, me: str, conversation_id: int, upto: Optional[int]) -> int:
    with tx(conn), conn.cursor() as cur:
        _member_or_404(cur, conversation_id, me)
        cur.execute("UPDATE proc.bp_conversation_participant SET last_read_message_id = GREATEST(last_read_message_id, "
                    "COALESCE(%s, (SELECT COALESCE(max(message_id),0) FROM proc.bp_message WHERE conversation_id=%s))) "
                    "WHERE conversation_id=%s AND user_subject=%s RETURNING last_read_message_id",
                    (upto, conversation_id, conversation_id, me))
        return cur.fetchone()[0]


def list_conversations(conn, me: str, limit: int = 50) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.conversation_id, c.type, c.title, c.linked_entity_type, c.linked_entity_id, c.last_message_at, p.muted, "
            "(SELECT count(*) FROM proc.bp_message m WHERE m.conversation_id=c.conversation_id AND "
            " m.message_id > p.last_read_message_id AND m.sender_subject<>%s AND m.deleted_at IS NULL) "
            "FROM proc.bp_conversation_participant p JOIN proc.bp_conversation c USING (conversation_id) "
            "WHERE p.user_subject=%s AND p.left_at IS NULL AND c.tenant_id=%s "
            "ORDER BY c.last_message_at DESC NULLS LAST, c.conversation_id DESC LIMIT %s", (me, me, TENANT, limit))
        rows = cur.fetchall()
        items = []
        for r in rows:
            parts = participants(cur, r[0])
            title = r[2] or (", ".join(p["name"] for p in parts if p["subject"] != me) if r[1] == "direct" else None)
            items.append({"id": r[0], "type": r[1], "title": title, "entity_type": r[3], "entity_id": r[4],
                          "last_message_at": r[5].isoformat() if r[5] else None, "muted": r[6], "unread": int(r[7]),
                          "participants": parts})
    return {"conversations": items, "unread_total": sum(i["unread"] for i in items)}


def get_conversation(conn, me: str, conversation_id: int) -> Dict[str, Any]:
    with conn.cursor() as cur:
        conv = _member_or_404(cur, conversation_id, me)
        return {**conv, "participants": participants(cur, conversation_id)}


# ------------------------------------------------------------------ recipient picker
def suggest_recipients(conn, me: str, q: Optional[str], conversation_id: Optional[int] = None, limit: int = 20) -> Dict[str, Any]:
    """Labelled groups. Tier 1 only (asserted facts): your manager, your reporting line, your reports,
    your department head, and everyone already in the conversation. Graph-derived tiers arrive with Phase 4.
    `search` finds anyone active in the tenant by name or email."""
    from services.people.hierarchy import HierarchyService
    from services.people.stores import PgPeopleStore
    store = PgPeopleStore(conn)
    mine = store.get(TENANT, me)
    groups: List[Dict[str, Any]] = []
    seen = {me}

    def add(label: str, reason: str, people: Iterable[Person]) -> None:
        items = [{"subject": p.subject, "name": p.display_name or p.subject, "reason": reason}
                 for p in people if p.is_active and p.subject not in seen]
        for i in items:
            seen.add(i["subject"])
        if items:
            groups.append({"label": label, "tier": "asserted", "people": items})

    if mine:
        h = HierarchyService(store)
        mgr = h.get_manager(mine)
        add("Your manager", "You report to them", [mgr] if mgr else [])
        add("Your reporting line", "Above you in the organisation", h.get_chain_of_command(mine)[1:])
        add("Your department head", "Heads your department", [x for x in [h.get_department_head(mine)] if x])
        add("Your direct reports", "They report to you", h.get_direct_reports(mine))
        if conversation_id:
            with conn.cursor() as cur:
                if is_participant(cur, conversation_id, me):
                    ps = [store.get(TENANT, p["subject"]) for p in participants(cur, conversation_id)]
                    add("Already in this conversation", "Taking part", [p for p in ps if p])
    found: List[Dict[str, Any]] = []
    if q and q.strip():
        like = f"%{q.strip().lower()}%"
        with conn.cursor() as cur:
            cur.execute("SELECT subject, COALESCE(display_name, subject) FROM proc.bp_person WHERE tenant_id=%s AND is_active "
                        "AND subject<>%s AND (lower(display_name) LIKE %s OR lower(email) LIKE %s) ORDER BY display_name LIMIT %s",
                        (TENANT, me, like, like, limit))
            found = [{"subject": s, "name": n, "reason": "Matches your search"} for s, n in cur.fetchall()]
    return {"suggested": groups, "search": found}
