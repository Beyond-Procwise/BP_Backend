"""Review API for the extraction feedback loop (propose-only).

Humans list pending per-vendor hint proposals, inspect the evidence, and
approve/reject them. Approve applies a versioned bp_prompt hint (live on reload);
reject records the reason so it is not re-proposed. A manual run endpoint lets an
operator trigger the proposer on demand.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from src.services.db import get_conn
from src.services.extraction_feedback import apply, proposer

log = logging.getLogger(__name__)

router = APIRouter(prefix="/extraction", tags=["Extraction Feedback"])

_COLS = (
    "proposal_id, created_date, doc_type, vendor_key, field_name, evidence, "
    "proposed_hint, rationale, status, reviewed_by, reviewed_date, review_reason, "
    "resulting_prompt_id"
)


class ReviewBody(BaseModel):
    """What is being decided. `approver` is NOT who decided it.

    It defaulted to the literal "api", which was written into proc.bp_prompt --
    the governance table whose prompts override the code -- as the hint's
    author. It stays on the model because clients send it; the approver is the
    token (see `_approver`).
    """

    approver: str = "api"
    reason: str | None = None


def _approver(principal) -> str | None:
    """The token, never `body.approver`. None when there is no principal."""

    return getattr(principal, "subject", None) or None


def _rows(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


@router.get("/proposals")
def list_proposals(status: str = "pending", limit: int = 100):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM proc.bp_extraction_hint_proposal "
            "WHERE (%s = 'all' OR status = %s) ORDER BY created_date DESC LIMIT %s",
            (status, status, limit),
        )
        proposals = _rows(cur)
    return {"count": len(proposals), "proposals": proposals}


@router.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: int):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM proc.bp_extraction_hint_proposal WHERE proposal_id = %s",
            (proposal_id,),
        )
        rows = _rows(cur)
    if not rows:
        raise HTTPException(status_code=404, detail=f"proposal {proposal_id} not found")
    return rows[0]


@router.post("/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: int, body: ReviewBody, principal=Depends(require_user)):
    try:
        return apply.approve(proposal_id, _approver(principal))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/proposals/{proposal_id}/reject")
def reject_proposal(proposal_id: int, body: ReviewBody, principal=Depends(require_user)):
    try:
        apply.reject(proposal_id, _approver(principal), body.reason)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "rejected", "proposal_id": proposal_id}


@router.post("/proposals/run")
def run_proposer(window_days: int = 14, draft: bool = True,
                 principal=Depends(require_user)):
    """Trigger the proposer on demand (ops/validation)."""
    ids = proposer.propose_all(window_days=window_days, draft=draft)
    return {"created": ids, "count": len(ids)}
