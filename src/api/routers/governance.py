"""Governed-reasoning API — AgentNick reasons with policy/prompt engines as tools."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Governed Reasoning"])


class GovernBody(BaseModel):
    task: str
    agent: str | None = None


@router.post("/govern")
def govern(body: GovernBody, principal=Depends(require_user)):
    # `body.agent` is the agent the task is governed AS -- it scopes which
    # policies apply and is recorded as the audit row's agent. It names an
    # agent, not a person, so it is not replaced by the principal.
    if os.getenv("GOVERNED_REASONING_ENABLED", "1") in ("0", "false", "False"):
        raise HTTPException(status_code=403, detail="governed reasoning disabled")
    if not body.task or not body.task.strip():
        raise HTTPException(status_code=400, detail="task is required")
    from src.services.governance_tools.governed_reasoning import govern as _govern
    return _govern(body.task, body.agent)
