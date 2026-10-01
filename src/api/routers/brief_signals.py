"""GET /spendiq/brief-signals — the figures Today's brief cannot derive itself.

The brief on the Procurement Home front door composes its readings from what
that page already fetches, plus four signals it had no source for and shipped
as a hard-coded MOCK_BRIEF_SIGNALS literal. This serves those four for real.

Read-only, and deliberately thin: everything it knows is in
``services/brief_signals_service``. The payload mirrors the shape the readings
registry already consumes, so wiring it up is a swap rather than a rewrite.

A signal the corpus cannot ground is absent from the response. ``sources`` says
which, so a short brief can be told apart from a broken one.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_user
from src.services import brief_signals_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Today's brief"])


@router.get("/brief-signals", summary="Signals for Today's brief, grounded or absent")
def get_brief_signals(principal: Any = Depends(require_user)) -> dict:
    """Every brief signal this corpus can ground, for this caller.

    The caller matters: ``myRequests`` is the demands THIS person raised, so it
    is resolved from the token and never from a query parameter. With auth off
    there is no principal, and that signal is reported unavailable rather than
    answered with everybody's requests.
    """
    try:
        return brief_signals_service.build_brief_signals(
            username=getattr(principal, "username", None),
            email=getattr(principal, "email", None),
            subject=getattr(principal, "subject", None),
        )
    except Exception as exc:
        # Per-signal failures are already isolated inside the service and come
        # back in `sources`; reaching here means something structural (no
        # database, say), which is a 503 and not an empty brief.
        logger.exception("brief-signals failed")
        raise HTTPException(status_code=503, detail="brief signals unavailable") from exc
