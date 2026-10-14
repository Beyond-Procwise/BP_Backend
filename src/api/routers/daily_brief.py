"""GET /spendiq/brief — the daily brief for whoever is signed in.

Today's brief was composed in the browser, which guessed the reader's view from
the NAME of their role. This is the server deciding instead: the reader comes
from the token, the view and every cut-off from the saved settings, and the
records from the reader's own buyer assignment. A chief procurement officer and
a buyer calling the same address get different briefs, and neither can ask for
the other's by changing a parameter -- there are none.

The settings are read and saved here too. Saving is ``configure``: it changes
what every person is told on Home.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services.daily_brief import service
from src.services.daily_brief.settings import BriefSettingsUnavailable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Daily brief"])

_AGENT = "DailyBriefRouter"


@router.get("/brief", summary="The daily brief for the signed-in reader")
def get_brief(principal: Any = Depends(require_user)) -> dict:
    try:
        return service.build(principal)
    except BriefSettingsUnavailable as exc:
        # No view is guessed for anybody: a brief built on settings that could
        # not be read would be the wrong person's brief, served with confidence.
        logger.error("daily brief refused: %s", exc)
        raise HTTPException(status_code=503, detail="the daily brief settings are unavailable") from exc
    except Exception as exc:
        logger.exception("daily brief failed")
        raise HTTPException(status_code=503, detail="the daily brief is unavailable") from exc


class SettingsBody(BaseModel):
    rules: Dict[str, Any] = Field(..., description="The settings to change, by name. Each one given replaces the saved one.")
    version: Optional[int] = Field(None, description="The version the screen loaded; a save against an older one is refused.")


@router.get("/brief/settings", summary="The daily brief settings, and what may be chosen")
def get_settings(principal: Any = Depends(require_user)) -> dict:
    gate("brief.settings.write", principal, agent=_AGENT, context={"op": "read"})
    try:
        return service.read_settings()
    except BriefSettingsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/brief/settings", summary="Change the daily brief settings")
def patch_settings(body: SettingsBody, principal: Any = Depends(require_user)) -> dict:
    gate("brief.settings.write", principal, agent=_AGENT,
         context={"op": "save", "settings": sorted(body.rules)})
    by = (getattr(principal, "email", None) or getattr(principal, "username", None)
          or getattr(principal, "subject", None) or "unknown")
    try:
        return service.save_settings(body.rules, by=str(by), expected_version=body.version)
    except service.SettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BriefSettingsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
