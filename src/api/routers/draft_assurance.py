"""What a reviewer needs to trust an email draft, and the events they raise about it.

    GET  /drafts/{unique_id}/assurance             the brief, facts, conflicts, flags, judge, accountability
    POST /drafts/{unique_id}/assumptions/confirm   confirm / edit / reject each assumption
    POST /drafts/{unique_id}/preflight             would a send be blocked by a fact that moved?
    POST /drafts/{unique_id}/abandon               the draft was closed without being sent

The person is ``principal.subject`` and nothing else: no body field names who confirmed or
abandoned, and there is no fallback when the principal is absent (see approvals.py, which closed
exactly that forgery). Responses carry human labels and row ids, never an internal table or column
name.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from src.services import guardrail, rbac
from src.services.draft_assurance import capture
from src.services.draft_assurance.send import recheck_for_send

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/drafts", tags=["Draft assurance"])

_ACTION, _CLASS = "email.draft", "write"       # reviewing a draft is part of drafting it


class Confirmation(BaseModel):
    id: str
    action: str
    value: Optional[Any] = None


class ConfirmRequest(BaseModel):
    confirmations: List[Confirmation]


class AbandonRequest(BaseModel):
    reason: Optional[str] = None


def _subject(principal: Any) -> str:
    who = str(getattr(principal, "subject", "") or "").strip()
    if not who:
        raise HTTPException(status_code=401, detail="this action must name an authenticated person")
    return who


def _authorize(principal: Any, unique_id: str) -> None:
    decision = guardrail.authorize(_ACTION, _CLASS, principal, {"unique_id": unique_id},
                                   policy_engine=rbac.policy_engine())
    if not decision.allowed:
        raise HTTPException(status_code=403, detail="you are not permitted to review this draft")


def _authorize_read(principal: Any, unique_id: str) -> None:
    """Name the person, then ask the read gate. A blank person is refused before anything is asked."""
    _subject(principal)
    decision = guardrail.authorize("email.draft.read", "read", principal, {"unique_id": unique_id},
                                   policy_engine=rbac.policy_engine())
    if not decision.allowed:
        raise HTTPException(status_code=403, detail="you are not permitted to view this draft")


def _conn():
    from src.services.db import get_conn
    return get_conn()


def _reviewer(conn: Any, unique_id: str) -> Optional[str]:
    try:
        from src.services import approval_store
        approval = approval_store.find_dispatch_approval(rfq_id=None, workflow_id=None, unique_id=unique_id, conn=conn)
        return (approval or {}).get("actioned_by")
    except Exception:  # noqa: BLE001
        return None


def _view(conn: Any, unique_id: str) -> Dict[str, Any]:
    raw = capture.load_raw(conn, unique_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="no assured draft with that id")
    return capture.to_view(raw, reviewed_by=_reviewer(conn, unique_id))


@router.get("/{unique_id}/assurance")
def get_assurance(unique_id: str, principal=Depends(require_user)) -> Dict[str, Any]:
    _authorize_read(principal, unique_id)
    with _conn() as conn:
        return _view(conn, unique_id)


@router.post("/{unique_id}/assumptions/confirm")
def confirm(unique_id: str, body: ConfirmRequest, principal=Depends(require_user)) -> Dict[str, Any]:
    by = _subject(principal)
    _authorize(principal, unique_id)
    with _conn() as conn:
        result = capture.confirm_assumptions(conn, unique_id, [c.dict() for c in body.confirmations], by)
        if not result.get("ok"):
            status = 404 if result.get("error") == "no such draft" else 422
            raise HTTPException(status_code=status, detail=result.get("error"))
        return _view(conn, unique_id)


@router.post("/{unique_id}/preflight")
def preflight(unique_id: str, principal=Depends(require_user)) -> Dict[str, Any]:
    _authorize_read(principal, unique_id)
    with _conn() as conn:
        raw = capture.load_raw(conn, unique_id)
        if raw is None:
            raise HTTPException(status_code=404, detail="no assured draft with that id")
        draft = {"assurance": {"family_id": raw["family_id"], "mode": raw.get("mode"), "facts": raw.get("facts") or {}}}
        result = recheck_for_send(conn, draft, rbac.policy_engine())
        facts = raw.get("facts") or {}
        changed = [{"fact": c["fact"], "label": (facts.get(c["fact"]) or {}).get("label") or c["fact"],
                    "was": c.get("was"), "now": c.get("now")} for c in result.get("changed") or []]
        ready = capture.readiness(conn, unique_id)
    return {"ok": result.get("checked", False) and not changed and ready.get("ready") is True,
            "checked": bool(result.get("checked")), "mode": result.get("mode") or raw.get("mode"),
            "changed": changed, "ready": ready.get("ready"),
            **({"reason": result["reason"]} if result.get("reason") else {})}


@router.post("/{unique_id}/abandon")
def abandon(unique_id: str, body: AbandonRequest, principal=Depends(require_user)) -> Dict[str, Any]:
    by = _subject(principal)
    with _conn() as conn:
        outcome = capture.record_abandoned(conn, unique_id, by, body.reason)
    if outcome is None:
        raise HTTPException(status_code=409, detail="nothing to abandon: the draft is unknown or was already sent")
    return {"ok": True}
