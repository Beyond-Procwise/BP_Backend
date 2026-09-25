"""/value — recording the money a finding or opportunity actually produced.
Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md §4"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.auth import require_user
from src.services import value_ledger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/value", tags=["Value ledger"])

_STATUS = {"finding_already_moved": 409, "no_open_claim": 409, "already_corrected": 409,
           "not_current": 409, "not_found": 404}


def _actor(principal: Any) -> str:
    """Who recorded it — from the token, never from the body."""
    subject = str(getattr(principal, "subject", "") or "").strip()
    if not subject:
        raise HTTPException(status_code=401, detail="no authenticated subject")
    return subject


def _call(fn, *args, **kwargs):
    # A dict `detail=` raised via HTTPException is flattened to its str() repr by the
    # app-wide `_safe_http_exception` handler (src/api/main.py), which only special-cases
    # a detail that is already a string. Returning a JSONResponse directly instead keeps
    # the `{"error": <code>, "message": <str>}` shape the brief specifies intact — it
    # still passes through OutputSafetyMiddleware like any other JSON body.
    try:
        return fn(*args, **kwargs)
    except value_ledger.LedgerError as exc:
        return JSONResponse(status_code=_STATUS.get(exc.code, 422),
                            content={"detail": {"error": exc.code, "message": str(exc)}})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("value ledger write failed")
        return JSONResponse(status_code=500,
                            content={"detail": {"error": "write_failed", "message": str(exc)}})


class OutcomeBody(BaseModel):
    outcome: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    valid_from: Optional[str] = None
    note: Optional[str] = None


class SettleBody(BaseModel):
    outcome: str
    amount: Optional[str] = None
    currency: Optional[str] = None
    evidence_ref: Optional[str] = None
    valid_from: Optional[str] = None
    note: Optional[str] = None


class RealiseBody(BaseModel):
    amount: str
    currency: str
    valid_from: Optional[str] = None
    evidence_ref: Optional[str] = None
    note: Optional[str] = None


class CorrectBody(BaseModel):
    amount: str
    currency: str
    note: str
    evidence_ref: Optional[str] = None


@router.post("/findings/{discrepancy_id}/outcome", summary="Close a money finding with what happened to it")
def post_outcome(discrepancy_id: int, body: OutcomeBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.record_finding_outcome, discrepancy_id, body.outcome, body.amount,
                 body.currency, actor=_actor(principal), valid_from=body.valid_from, note=body.note)


@router.post("/findings/{discrepancy_id}/settle", summary="Credit received, or the claim dropped")
def post_settle(discrepancy_id: int, body: SettleBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.settle_claim, discrepancy_id, body.outcome, body.amount,
                 body.currency, actor=_actor(principal), evidence_ref=body.evidence_ref,
                 valid_from=body.valid_from, note=body.note)


@router.post("/opportunities/{opportunity_id}/realise", summary="Mark an opportunity's saving realised")
def post_realise(opportunity_id: str, body: RealiseBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.realise_opportunity, opportunity_id, body.amount, body.currency,
                 actor=_actor(principal), valid_from=body.valid_from,
                 evidence_ref=body.evidence_ref, note=body.note)


@router.post("/outcomes/{outcome_id}/correct", summary="Correct a recorded figure (a new row)")
def post_correct(outcome_id: int, body: CorrectBody, principal=Depends(require_user)) -> dict:
    return _call(value_ledger.correct_outcome, outcome_id, body.amount, body.currency,
                 actor=_actor(principal), note=body.note, evidence_ref=body.evidence_ref)


@router.get("/findings/{discrepancy_id}/outcomes", summary="A finding's outcome history and suggested figure")
def get_outcomes(discrepancy_id: int) -> dict:
    return _call(value_ledger.finding_outcomes, discrepancy_id)
