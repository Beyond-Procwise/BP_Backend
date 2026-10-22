"""Who is working what: hand a finding to a named person (D-73, second half).

  POST   /work/{entity_type}/{entity_id}/assign {to_subject, note?}  hand it on (yourself = claim)
  DELETE /work/{entity_type}/{entity_id}/assign                      nobody works it
  GET    /work/assignments?entity_type=finding&ids=1,2,3             the active assignments (names included)
  GET    /work/mine                                                  what is with the signed-in person

Handing on is a triage write (finding.triage); reads need only a signed-in user. The receiver is told
with a task.assigned notification that says what the finding is and links to it.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.db import get_conn
from services.people import work as svc
from services.people.admin import PeopleError

router = APIRouter(prefix="/work", tags=["work"])


class AssignBody(BaseModel):
    to_subject: str = Field(max_length=256)
    note: Optional[str] = Field(default=None, max_length=svc.MAX_NOTE)


def _me(principal: Any) -> str:
    sub = str(getattr(principal, "subject", "") or "").strip()
    if not sub:
        raise HTTPException(status_code=401, detail="sign in first")
    return sub


def _run(fn, *args, **kwargs):
    try:
        with get_conn() as conn:
            return fn(conn, *args, **kwargs)
    except PeopleError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)


@router.post("/{entity_type}/{entity_id}/assign")
def assign(entity_type: str, entity_id: str, body: AssignBody, principal=Depends(require_user)):
    me = _me(principal)
    gate("finding.triage", principal, agent="work_api", context={"entity_type": entity_type, "entity_id": entity_id})
    return _run(svc.assign, me, entity_type, entity_id, body.to_subject, body.note)


@router.delete("/{entity_type}/{entity_id}/assign")
def release(entity_type: str, entity_id: str, principal=Depends(require_user)):
    me = _me(principal)
    gate("finding.triage", principal, agent="work_api", context={"entity_type": entity_type, "entity_id": entity_id})
    return {"released": _run(svc.release, me, entity_type, entity_id)}


@router.get("/assignments")
def assignments(entity_type: str = Query(default="finding"), ids: Optional[str] = Query(default=None),
                principal=Depends(require_user)):
    _me(principal)
    wanted = [s for s in (ids or "").split(",") if s.strip()] if ids else None
    return {"assignments": _run(svc.active, entity_type, wanted)}


@router.get("/mine")
def mine(principal=Depends(require_user)):
    me = _me(principal)
    return {"assignments": _run(svc.mine, me)}
