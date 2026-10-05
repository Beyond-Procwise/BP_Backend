"""Extraction for the Demand Intake conversation.

The screen owns the questions; this endpoint owns the one model call. It takes the VALUES to fill
the governed prompt with — this tenant's categories, the field list from the intake
configuration, what is already known, the question just asked, and the requester's own words —
renders `demand_intake_extract` from proc.bp_prompt, and returns the field values the model read
out of the text.

It does not take a prompt. The screen used to build one and hand it over, which meant the
instruction travelled from the client; now the instruction is governed and the client sends only
data. If the governed prompt is missing this answers 503 rather than running something else, and
the conversation carries on asking its next question — the browser reads the text itself and
marks every value it finds as read from the description rather than as the model's.

Nothing here persists. The record is written by the demand endpoints in the gateway, over
proc.bp_demand — NOT proc.bp_demand_tprm, which does not exist in either database (checked in
bp_testdb and bp_sqldb on 2026-10-04). There must not be a second demand store.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.agents.demand_intake_agent import DemandIntakeUnavailable, extract_fields

from api.auth import require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demand", tags=["Demand intake"])

# One turn of a conversation, not a document. The reference deck's longest intake paragraph is a
# few hundred characters; 20k is room for someone pasting a requirement in, and a refusal above
# it is cheaper than a model call that will time out.
_MAX_TEXT = 20_000


class ExtractBody(BaseModel):
    """`context` is data, never an instruction: the field list and the categories come from the
    intake configuration, the text is what the requester typed."""

    context: Dict[str, Any]


def get_agent_nick(request: Request):
    nick = getattr(request.app.state, "agent_nick", None)
    if not nick:
        # 503, not 500: the model host is a dependency that can be down, and the screen has a
        # path that does not need it.
        raise HTTPException(status_code=503, detail="the local model is not available")
    return nick


@router.post("/intake/extract")
async def post_extract(body: ExtractBody, principal=Depends(require_user),
                       agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    context = dict(body.context or {})
    text = str(context.get("text") or "")
    if not text.strip():
        raise HTTPException(status_code=400,
                            detail="there is nothing to read: the request carried no text")
    if len(text) > _MAX_TEXT:
        raise HTTPException(status_code=413,
                            detail=f"that is longer than {_MAX_TEXT} characters; "
                                   "send one answer at a time")
    try:
        result = await run_in_threadpool(extract_fields, agent_nick, context)
    except DemandIntakeUnavailable as exc:
        # The reason is passed on rather than paraphrased: "extraction is not authorised because
        # its prompt is not installed" is actionable, and "could not extract" is not.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Fields and provenance only. The prompt is governed server-side precisely so that it is not
    # something a client holds, and echoing it would hand every caller the text to tamper with.
    return {"fields": result.get("fields") or {},
            "governed": bool(result.get("governed")),
            "failed": bool(result.get("failed"))}
