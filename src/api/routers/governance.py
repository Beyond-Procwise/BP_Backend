"""Governed-reasoning API — AgentNick reasons with policy/prompt engines as tools."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Governed Reasoning"])


class GovernBody(BaseModel):
    task: str
    agent: str | None = None


@router.post("/govern")
def govern(body: GovernBody):
    if os.getenv("GOVERNED_REASONING_ENABLED", "1") in ("0", "false", "False"):
        raise HTTPException(status_code=403, detail="governed reasoning disabled")
    if not body.task or not body.task.strip():
        raise HTTPException(status_code=400, detail="task is required")
    from src.services.governance_tools.governed_reasoning import govern as _govern
    return _govern(body.task, body.agent)
