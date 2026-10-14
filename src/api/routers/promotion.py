"""Confidence-gated _stg -> _trgt promotion driven by the linking engine.

POST /promotion/run — score each not-yet-promoted staged invoice/quote against
its parent PO and promote the ones whose extraction confidence and document
link score both pass. deal_id is left for the SQL trigger.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from src.services.link_proposals import (
    confirm_parent_link, propose_parent_links, reject_parent_link,
)
from src.services.linking_engine import (
    promote_ready, review_queue, approve_promotion, quote_chains, canonicalize_po_references,
)

from api.auth import require_user
from api.endpoint_gate import require as gate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/promotion", tags=["Promotion"])



def _reviewer(principal: Any) -> str | None:
    """Who signed this, which is the token and only the token.

    Every decision endpoint in this router used to accept `body["reviewer"]`,
    and one of them still preferred the principal but fell back to it. A
    fallback is the forgery taken conditionally: it is exactly the request a
    caller sends when they want somebody else's name on a decision that writes
    a reference into the financial record.

    None when there is no principal (ASK_AUTH_MODE=off). A decision recorded
    against nobody is honest; one recorded against a name that was typed is not.
    """

    return getattr(principal, "subject", None) or None

@router.post("/run", summary="Promote eligible _stg rows to _trgt via the linking engine")
def run_promotion(
    doc_type: Optional[str] = Query(None, description="invoice | quote (default: both)"),
    limit: Optional[int] = Query(None, description="max rows per doc type"),
    principal=Depends(require_user),
) -> dict[str, Any]:
    gate("document.promote", principal, agent="PromotionRouter",
         context={"doc_type": doc_type})
    doc_types = (doc_type,) if doc_type else ("invoice", "quote")
    if any(d not in ("invoice", "quote") for d in doc_types):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    try:
        return promote_ready(doc_types=doc_types, limit=limit)
    except Exception as exc:
        logger.exception("promotion run failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/canonicalize-po", summary="Align po_id across _trgt to the bare PO number")
def post_canonicalize_po(principal=Depends(require_user)) -> dict[str, Any]:
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
    principal=Depends(require_user),
) -> dict[str, Any]:
    gate("document.promote", principal, agent="PromotionRouter",
         context={"doc_type": doc_type, "doc_pk": doc_pk})
    if doc_type not in ("invoice", "quote"):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    # The reviewer is who signed in, not who the caller says signed in. The
    # body value survives only as a fallback while authentication is off --
    # this row is the record that a human released a document into _trgt.
    reviewer = _reviewer(principal)
    try:
        result = approve_promotion(doc_type, doc_pk,
                                   reviewer=reviewer, note=body.get("note"))
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
        run = propose_parent_links(doc_type=dt, min_score=min_score)
    except Exception as exc:
        logger.exception("link-proposals failed")
        raise HTTPException(status_code=500, detail=str(exc))
    # `considered` is not decoration: this pass proposes nothing at all on the
    # current corpus, and a bare empty list reads as "every document has an order"
    # when the truth is that 1,964 have none. The counts are what let a screen say
    # which of the two it is looking at.
    return {"count": len(run.proposals), "considered": run.considered,
            "items": [asdict(i) for i in run.proposals]}


def _decision_result(result: dict) -> dict:
    """One mapping for both decisions, so accepting and refusing a proposal cannot
    drift into reporting the same outcome differently."""
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=result["detail"])
    if result.get("status") == "refused":
        # The document's own state says no — it already names an order, or this one
        # was never proposed for it. That is a conflict with what the caller believes,
        # not a fault to retry.
        raise HTTPException(status_code=409, detail=result["detail"])
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["detail"])
    return result


@router.post("/link-proposals/{doc_type}/{doc_pk}/reject",
             summary="Human-reject a proposed parent order, so it is not proposed again")
def post_reject_link(
    doc_type: str,
    doc_pk: str,
    body: dict[str, Any] = Body(default_factory=dict),
    principal=Depends(require_user),
) -> dict[str, Any]:
    """Record that this order is NOT the parent of this document.

    A separate endpoint from the confirm rather than one taking a verb: confirming
    writes a reference onto a document and rejecting must never be able to, and a
    shared route with a parameter is how a client eventually sends the wrong one.

    It suppresses one pairing. The document may be proposed a different order next
    pass — "not this one" is not "this belongs nowhere".
    """
    if doc_type not in ("invoice", "quote"):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    po_id = (body.get("po_id") or "").strip()
    if not po_id:
        raise HTTPException(status_code=400, detail="po_id is required")
    try:
        result = reject_parent_link(doc_type, doc_pk, po_id,
                                    reviewer=_reviewer(principal), note=body.get("note"))
    except Exception as exc:
        logger.exception("reject link failed for %s %s", doc_type, doc_pk)
        raise HTTPException(status_code=500, detail=str(exc))
    return _decision_result(result)


@router.post("/link-proposals/{doc_type}/{doc_pk}/confirm",
             summary="Human-confirm a proposed parent order for a document")
def post_confirm_link(
    doc_type: str,
    doc_pk: str,
    body: dict[str, Any] = Body(default_factory=dict),
    principal=Depends(require_user),
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
                                     reviewer=_reviewer(principal), note=body.get("note"))
    except Exception as exc:
        logger.exception("confirm link failed for %s %s", doc_type, doc_pk)
        raise HTTPException(status_code=500, detail=str(exc))
    return _decision_result(result)
