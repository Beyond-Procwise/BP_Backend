"""Emails to colleagues: write one, see your pending ones, change, approve or decline.

Who you are is the email on your sign-in token, never a field in the request: a body that
says ``"author": "someone@else"`` is ignored, and only the author's own token can approve.
"""
from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from services import internal_email
from services.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mail/internal", tags=["Mail integration"])

_AGENT = "InternalEmailRouter"
_STATUS = {"not_found": 404, "not_author": 403, "changed": 409, "already_done": 409,
           "policy_off": 422, "not_internal": 422, "no_mailbox": 422}


def _conn():
    return get_conn()


def _me(principal: Any) -> str:
    """The caller's email from the token. Empty when the token carries none: such a caller
    is nobody's author, so every author-only action refuses them."""
    return str(getattr(principal, "email", None) or "").strip()


def _detail(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": message}})


def _refused(exc: internal_email.InternalEmailRefused) -> JSONResponse:
    return _detail(_STATUS.get(exc.code, 422), exc.code, exc.message)


def _view(row: dict) -> dict:
    out = dict(row)
    for k in ("approved_at", "sent_at", "created_at"):
        if out.get(k) is not None and hasattr(out[k], "isoformat"):
            out[k] = out[k].isoformat()
    return out


class CreateBody(BaseModel):
    kind: Literal["forward", "reply", "notice"]
    recipients: List[str] = Field(..., min_length=1)
    subject: str
    body: str
    source_ref: Optional[str] = None


class EditBody(BaseModel):
    recipients: List[str] = Field(..., min_length=1)
    subject: str
    body: str


def _run(fn, *args, **kwargs) -> Any:
    try:
        with _conn() as conn:
            return _view(fn(conn, *args, **kwargs))
    except internal_email.InternalEmailRefused as exc:
        return _refused(exc)
    except ValueError as exc:
        return _detail(422, "invalid", str(exc))


@router.post("", summary="Write an email to colleagues; it waits for your approval")
def create(body: CreateBody, principal: Any = Depends(require_user)) -> Any:
    gate("mail.internal.create", principal, agent=_AGENT, context={"op": "create", "kind": body.kind})
    return _run(internal_email.create, author=_me(principal), kind=body.kind,
                recipients=body.recipients, subject=body.subject, body=body.body,
                attachments=[], source_ref=body.source_ref)


@router.get("/pending", summary="Your emails to colleagues waiting for your approval")
def pending(principal: Any = Depends(require_user)) -> Any:
    me = _me(principal)
    if not me:
        return {"items": []}
    with _conn() as conn:
        return {"items": [_view(r) for r in internal_email.list_pending(conn, author=me)]}


@router.patch("/{internal_email_id}", summary="Change your pending email to colleagues")
def edit(internal_email_id: int, body: EditBody, principal: Any = Depends(require_user)) -> Any:
    gate("mail.internal.create", principal, agent=_AGENT,
         context={"op": "edit", "internal_email_id": internal_email_id})
    return _run(internal_email.edit, internal_email_id, author=_me(principal),
                subject=body.subject, body=body.body, recipients=body.recipients)


@router.post("/{internal_email_id}/approve", summary="Approve and send your email to colleagues")
def approve(internal_email_id: int, principal: Any = Depends(require_user)) -> Any:
    gate("mail.internal.approve", principal, agent=_AGENT,
         context={"op": "approve", "internal_email_id": internal_email_id})
    return _run(internal_email.approve_and_send, internal_email_id, approver=_me(principal))


@router.post("/{internal_email_id}/decline", summary="Decline your email to colleagues")
def decline(internal_email_id: int, principal: Any = Depends(require_user)) -> Any:
    gate("mail.internal.approve", principal, agent=_AGENT,
         context={"op": "decline", "internal_email_id": internal_email_id})
    return _run(internal_email.decline, internal_email_id, actor=_me(principal))
