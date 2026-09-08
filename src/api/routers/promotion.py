"""Confidence-gated _stg -> _trgt promotion driven by the linking engine.

POST /promotion/run — score each not-yet-promoted staged invoice/quote against
its parent PO and promote the ones whose extraction confidence and document
link score both pass. deal_id is left for the SQL trigger.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from src.services.link_proposals import confirm_parent_link, propose_parent_links
from src.services.linking_engine import (
    promote_ready, review_queue, approve_promotion, quote_chains, canonicalize_po_references,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/promotion", tags=["Promotion"])


@router.post("/run", summary="Promote eligible _stg rows to _trgt via the linking engine")
def run_promotion(
    doc_type: Optional[str] = Query(None, description="invoice | quote (default: both)"),
    limit: Optional[int] = Query(None, description="max rows per doc type"),
) -> dict[str, Any]:
    doc_types = (doc_type,) if doc_type else ("invoice", "quote")
    if any(d not in ("invoice", "quote") for d in doc_types):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    try:
        return promote_ready(doc_types=doc_types, limit=limit)
    except Exception as exc:
        logger.exception("promotion run failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/canonicalize-po", summary="Align po_id across _trgt to the bare PO number")
def post_canonicalize_po() -> dict[str, Any]:
    try:
        return canonicalize_po_references()
    except Exception as exc:
        logger.exception("canonicalize-po failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/quote-chains", summary="Quote-anchored view: each quote -> its PO -> invoices")
def get_quote_chains() -> dict[str, Any]:
    try:
        return quote_chains()
    except Exception as exc:
        logger.exception("quote-chains failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/review-queue", summary="Held documents in the review band, with gap reports")
def get_review_queue(
    doc_type: Optional[str] = Query(None, description="invoice | quote (default: both)"),
    min_score: Optional[float] = Query(None, description="F floor (default PROMOTE_REVIEW_MIN=65)"),
    all: bool = Query(False, description="return EVERY currently-held doc (any hold reason), "
                                          "not just the F-score review band"),
    deal_id: Optional[str] = Query(None, description="filter items to one deal_id"),
) -> dict[str, Any]:
    doc_types = (doc_type,) if doc_type else ("invoice", "quote")
    if any(d not in ("invoice", "quote") for d in doc_types):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    try:
        items = review_queue(doc_types=doc_types, min_score=min_score,
                             all_held=all, deal_id=deal_id)
    except Exception as exc:
        logger.exception("review-queue failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"count": len(items), "items": items}


@router.post("/review/{doc_type}/{doc_pk}/approve", summary="Human-approve a held doc into _trgt")
def post_approve(
    doc_type: str,
    doc_pk: str,
    body: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    if doc_type not in ("invoice", "quote"):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    try:
        result = approve_promotion(doc_type, doc_pk,
                                   reviewer=body.get("reviewer"), note=body.get("note"))
    except Exception as exc:
        logger.exception("approve failed for %s %s", doc_type, doc_pk)
        raise HTTPException(status_code=500, detail=str(exc))
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=result["detail"])
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["detail"])
    return result


@router.get("/link-proposals",
            summary="Proposed parent orders for documents that reference none")
def get_link_proposals(
    doc_type: Optional[str] = Query(None, description="invoice | quote (default: invoice)"),
    min_score: Optional[float] = Query(None, description="F floor (default PROPOSE_MIN_LINK_SCORE=65)"),
) -> dict[str, Any]:
    """Read-only. A document carrying no purchase-order reference is never scored
    by the promotion path, so it has never reached the review queue — its F is
    None and the queue is an F-score window.

    Each item is what the evidence proposes and how forced the proposal was:
    ``suggested`` where the resolution layer's margin separates this order from
    every alternative it was weighed against, ``contested`` where it does not and
    a person is choosing between near-equals. Nothing is written by this call.
    """
    dt = doc_type or "invoice"
    if dt not in ("invoice", "quote"):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    try:
        items = propose_parent_links(doc_type=dt, min_score=min_score)
    except Exception as exc:
        logger.exception("link-proposals failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"count": len(items), "items": [asdict(i) for i in items]}


@router.post("/link-proposals/{doc_type}/{doc_pk}/confirm",
             summary="Human-confirm a proposed parent order for a document")
def post_confirm_link(
    doc_type: str,
    doc_pk: str,
    body: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Accept one of the orders proposed for this document, and write the
    reference on the reviewer's authority. Only an order this engine proposed —
    the winner or one of the alternatives shown with it — can be confirmed."""
    if doc_type not in ("invoice", "quote"):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    po_id = (body.get("po_id") or "").strip()
    if not po_id:
        raise HTTPException(status_code=400, detail="po_id is required")
    try:
        result = confirm_parent_link(doc_type, doc_pk, po_id,
                                     reviewer=body.get("reviewer"), note=body.get("note"))
    except Exception as exc:
        logger.exception("confirm link failed for %s %s", doc_type, doc_pk)
        raise HTTPException(status_code=500, detail=str(exc))
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=result["detail"])
    if result.get("status") == "refused":
        raise HTTPException(status_code=409, detail=result["detail"])
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["detail"])
    return result
