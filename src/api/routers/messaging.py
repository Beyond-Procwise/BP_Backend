"""Messaging between people: direct, group and per-item threads.

POST /messages/direct                          {to}                      open (or find) a 1:1 conversation
POST /messages/groups                          {title, members[]}
POST /messages/entity-thread                   {entity_type, entity_id}  discussion beside a PO/contract/supplier/...
GET  /messages/conversations                   mine, with unread counts
GET  /messages/conversations/{id}              details + participants
GET  /messages/conversations/{id}/messages     newest first; ?before_id=&limit=
POST /messages/conversations/{id}/messages     {body, mentions?, reply_to_id?, attachments?}
POST /messages/conversations/{id}/read         {message_id?}
PUT  /messages/conversations/{id}/mute         {muted}
POST /messages/conversations/{id}/participants {subject}
POST /messages/conversations/{id}/leave
PUT/DELETE /messages/{message_id}              edit / delete my own message
GET  /messages/suggest-recipients              ?q=&conversation_id=   labelled groups for the "Send to" picker
Every route works on the signed-in person's own conversations only; a conversation you are not in is a 404.
"""
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_user
from services.db import get_conn
from services.people import messaging as svc
from services.people import realtime
from services.people.admin import PeopleError

router = APIRouter(prefix="/messages", tags=["messaging"])


def _require_member(principal: Any) -> str:
    """Messaging is for signed-in people; every query below is keyed on this subject."""
    subject = getattr(principal, "subject", None)
    if not subject:
        raise HTTPException(status_code=401, detail="sign in to use messages")
    return str(subject)


def _run(fn, *args, **kwargs):
    try:
        with get_conn() as conn:
            return fn(conn, *args, **kwargs)
    except PeopleError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)


def _picker(conn, me, q, conversation_id):
    """Tier 1 groups from the org chart; tiers 2-3 from the graph only when the tenant switched them on.
    Every person shown is logged (who, which tier) so acceptance can be measured."""
    out = svc.suggest_recipients(conn, me, q, conversation_id)
    try:
        from services.people.admin import TENANT
        from services.people.graph import activity, queries, service
        if service.tiers_enabled(conn):
            key = None
            if conversation_id:
                conv = svc.get_conversation(conn, me, conversation_id)
                key = f"{conv['entity_type']}:{conv['entity_id']}" if conv.get("entity_type") else None
            extra = [r for r in queries.suggest_recipients(service.get_store(), TENANT, me, entity_key=key, graph_tiers=True)
                     if r["tier"] >= 2]
            known = {p["subject"] for g in out["suggested"] for p in g["people"]}
            for tier, label in ((2, "Have decided on this before"), (3, "You often work with")):
                people = [{"subject": r["subject"], "name": r["name"], "reason": r["reason"]} for r in extra
                          if r["tier"] == tier and r["subject"] not in known]
                if people:
                    out["suggested"].append({"label": label, "tier": "derived" if tier == 3 else "asserted", "people": people,
                                             "advisory": tier == 3})
        with conn.cursor() as cur:
            for i, g in enumerate(out["suggested"]):
                tier = "1" if g["label"] in _TIER1 else ("2" if g["label"].startswith("Have decided") else "3")
                for p in g["people"]:
                    activity.record(cur, "suggestion.shown", me, target_user_id=p["subject"], tier=tier)
    except Exception:
        pass                                   # suggestions never fail because logging or the graph did
    return out


_TIER1 = {"Your manager", "Your reporting line", "Your department head", "Your direct reports", "Already in this conversation"}


class FeedbackBody(BaseModel):
    subject: str
    tier: str
    outcome: str            # accepted | dismissed


class DirectBody(BaseModel):
    to: str


class GroupBody(BaseModel):
    title: str
    members: List[str]


class ThreadBody(BaseModel):
    entity_type: str
    entity_id: str
    title: Optional[str] = None


class SendBody(BaseModel):
    body: str
    mentions: Optional[List[str]] = None
    reply_to_id: Optional[int] = None
    attachments: Optional[List[dict]] = None


class EditBody(BaseModel):
    body: str


class ReadBody(BaseModel):
    message_id: Optional[int] = None


class MuteBody(BaseModel):
    muted: bool


class MemberBody(BaseModel):
    subject: str


@router.post("/direct", summary="Open (or find) a conversation with one person")
def open_direct(body: DirectBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.open_direct, me, body.to)


@router.post("/groups", summary="Start a group conversation")
def open_group(body: GroupBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.open_group, me, body.title, body.members)


@router.post("/entity-thread", summary="Start (or return) the discussion on a PO, contract, supplier...")
def open_thread(body: ThreadBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.open_entity_thread, me, body.entity_type, body.entity_id, body.title)


@router.get("/conversations", summary="My conversations with unread counts")
def conversations(limit: int = Query(50, ge=1, le=100), principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.list_conversations, me, limit)


@router.get("/suggest-recipients", summary="Who to send to: your manager, reporting line, reports, plus search")
def suggest(q: Optional[str] = None, conversation_id: Optional[int] = None,
            principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(_picker, me, q, conversation_id)


@router.get("/conversations/{conversation_id}", summary="One conversation")
def conversation(conversation_id: int, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.get_conversation, me, conversation_id)


@router.get("/conversations/{conversation_id}/messages", summary="Messages, newest first")
def messages(conversation_id: int, before_id: Optional[int] = None, limit: int = Query(30, ge=1, le=100),
             principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.list_messages, me, conversation_id, before_id, limit)


@router.post("/conversations/{conversation_id}/messages", summary="Send a message")
def send(conversation_id: int, body: SendBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    out = _run(svc.send, me, conversation_id, body.body, body.mentions, body.reply_to_id, body.attachments)
    realtime.push(out.pop("signal"), {"type": "message", "conversation_id": conversation_id, "message_id": out["id"]})
    return out


@router.post("/conversations/{conversation_id}/read", summary="Mark read up to a message (default: all)")
def read(conversation_id: int, body: ReadBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return {"last_read_message_id": _run(svc.mark_read, me, conversation_id, body.message_id)}


@router.put("/conversations/{conversation_id}/mute", summary="Silence (or restore) ordinary messages")
def mute(conversation_id: int, body: MuteBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    _run(svc.set_muted, me, conversation_id, body.muted)
    return {"muted": body.muted}


@router.post("/conversations/{conversation_id}/participants", summary="Add someone to a group or thread")
def add_member(conversation_id: int, body: MemberBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    out = _run(svc.add_participant, me, conversation_id, body.subject)
    realtime.push([body.subject], {"type": "conversation", "conversation_id": conversation_id})
    return out


@router.post("/conversations/{conversation_id}/leave", summary="Leave a conversation")
def leave(conversation_id: int, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    _run(svc.leave, me, conversation_id)
    return {"left": True}


@router.put("/{message_id}", summary="Edit one of my messages")
def edit(message_id: int, body: EditBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.edit, me, message_id, body.body)


@router.delete("/{message_id}", summary="Delete one of my messages")
def delete(message_id: int, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    return _run(svc.delete, me, message_id)


@router.post("/suggestions/feedback", summary="Tell the learning loop whether a suggested person was used")
def suggestion_feedback(body: FeedbackBody, principal=Depends(require_user)) -> dict[str, Any]:
    me = _require_member(principal)
    if body.outcome not in ("accepted", "dismissed") or body.tier not in ("1", "2", "3"):
        raise HTTPException(status_code=422, detail="outcome is accepted or dismissed; tier is 1, 2 or 3")
    from services.people.graph import activity
    def run(conn):
        with conn.cursor() as cur:
            return activity.record(cur, f"suggestion.{body.outcome}", me, target_user_id=body.subject, tier=body.tier)
    return {"recorded": bool(_run(run))}
