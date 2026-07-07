"""Supplier match-review API.

Humans confirm or reject close-call supplier-identity decisions. Confirm merges
the extracted variant into the candidate (alias); reject records it as a distinct
supplier. The document's literal extracted value is never changed — only the
canonical supplier_id link is governed.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as SR

log = logging.getLogger(__name__)

router = APIRouter(prefix="/suppliers", tags=["Supplier Review"])

_COLS = (
    "review_id, created_date, extracted_name, decision, chosen_supplier_id, "
    "candidate_supplier_id, candidate_supplier_name, score, doc_type, doc_pk, "
    "status, reviewed_by, reviewed_date"
)


class ReviewBody(BaseModel):
    reviewer: str = "api"


@router.get("/reviews")
def list_reviews(status: str = "pending", limit: int = 100):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM proc.bp_supplier_review "
            "WHERE (%s = 'all' OR status = %s) ORDER BY created_date DESC LIMIT %s",
            (status, status, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"count": len(rows), "reviews": rows}


@router.post("/reviews/sweep")
def sweep_duplicates(min_score: float | None = None):
    """Scan existing suppliers for likely duplicates and flag them for review."""
    with get_conn() as c:
        return SR.sweep_supplier_duplicates(c, min_score=min_score)


@router.post("/reviews/{review_id}/confirm")
def confirm_review(review_id: int, body: ReviewBody):
    """The extracted name IS the candidate supplier — merge (alias to canonical)."""
    with get_conn() as c:
        try:
            return SR.confirm_review(review_id, body.reviewer, c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.post("/reviews/{review_id}/reject")
def reject_review(review_id: int, body: ReviewBody):
    """The extracted name is a DISTINCT supplier — keep it separate (alias to its own id)."""
    with get_conn() as c:
        try:
            return SR.reject_review(review_id, body.reviewer, c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
