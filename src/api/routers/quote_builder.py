"""The Quote builder's API (UI BACKEND_GAPS.md, "Quote builder").

The paths are the ones the screen was built against; the record behind them is the
sell-side quote, so review, approval, numbering, the customer page and sending are the
/sales ones and are not repeated here.

Every read on this router returns the seller's cost and markup with the quote, so every
read asks margin.read and needs a STATED permit, exactly as GET /sales/quotes/{id}/internal
does. Who acted is the token's subject; no body has a field for it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.services import entity_profile
from src.services.sell_side import builder, pricing_checks, quote_agent, quotes
from src.services.sell_side._db import transactional_conn as get_conn

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.routers.sales import _require_margin_permit, _subject
from api.sell_side_http import http_errors, money_json

router = APIRouter(tags=["Quote builder"])
_AGENT = "QuoteBuilderRouter"


class QuoteDraft(BaseModel):
    """The quote as the screen holds it. The screen sends its whole object back, so
    fields that are not the drafter's to write (id, status, number, version, conditions,
    checks, totals) are accepted and ignored rather than refused."""
    model_config = ConfigDict(extra="ignore")

    currency: str
    validUntil: str
    issuedAt: Optional[str] = None
    category: Optional[str] = None
    preparedBy: Optional[str] = None
    customer: Optional[Dict[str, Any]] = None
    scopeHtml: Optional[str] = None
    assumptionsHtml: Optional[str] = None
    terms: Optional[Dict[str, Any]] = None
    lines: List[Dict[str, Any]] = Field(default_factory=list)
    dealId: Optional[str] = None
    # What the last read returned as `revision`. Sent back, it stops a save from
    # silently undoing somebody else's later save.
    revision: Optional[str] = None


class ApproveBody(BaseModel):
    conditions: List[str] = Field(default_factory=list)


class RejectBody(BaseModel):
    reason: str


def _with_checks(quote: Dict[str, Any]) -> Dict[str, Any]:
    return {**quote, "checks": pricing_checks.for_quote(quote.get("category"))}


def _draft(body: QuoteDraft) -> Dict[str, Any]:
    return body.model_dump(exclude={"dealId", "revision"})


# --- reads ------------------------------------------------------------------

@router.get("/spendiq/deals/{deal_id}/quotes")
def deal_quotes(deal_id: str, principal=Depends(require_user)):
    """The deal's quotes, latest first."""
    _require_margin_permit(principal, {"deal_id": deal_id})
    with http_errors(), get_conn() as c:
        return money_json([_with_checks(q) for q in builder.list_for_deal(c, deal_id)])


@router.get("/spendiq/quotes/unlinked")
def my_unlinked_quotes(principal=Depends(require_user)):
    """The caller's own quotes that are not on a deal yet."""
    _require_margin_permit(principal, {"scope": "unlinked"})
    with http_errors(), get_conn() as c:
        return money_json([_with_checks(q)
                           for q in builder.list_unlinked(c, _subject(principal))])


@router.get("/spendiq/quotes/review-queue")
def review_queue(principal=Depends(require_user)):
    """Every quote waiting for a reviewer, oldest first."""
    _require_margin_permit(principal, {"scope": "review-queue"})
    with http_errors(), get_conn() as c:
        return money_json([_with_checks(q) for q in builder.review_queue(c)])


@router.get("/spendiq/quotes/{quote_id}")
def get_quote(quote_id: int, principal=Depends(require_user)):
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(_with_checks(builder.get(c, quote_id)))


# --- drafting ---------------------------------------------------------------

@router.post("/spendiq/deals/{deal_id}/quotes")
def create_on_deal(deal_id: str, body: QuoteDraft, principal=Depends(require_user)):
    gate("sales_quote.create", principal, agent=_AGENT, context={"deal_id": deal_id})
    _require_margin_permit(principal, {"deal_id": deal_id})
    with http_errors(), get_conn() as c:
        return money_json(_with_checks(builder.create(
            c, deal_id=deal_id, draft=_draft(body), created_by=_subject(principal))))


@router.post("/spendiq/quotes")
def create_quote(body: QuoteDraft, principal=Depends(require_user)):
    """A quote started before its deal exists (`dealId: null`), or on a deal named in
    the body. It is linked before it is sent."""
    gate("sales_quote.create", principal, agent=_AGENT, context={"deal_id": body.dealId})
    _require_margin_permit(principal, {"deal_id": body.dealId})
    with http_errors(), get_conn() as c:
        return money_json(_with_checks(builder.create(
            c, deal_id=body.dealId, draft=_draft(body), created_by=_subject(principal))))


@router.put("/spendiq/quotes/{quote_id}")
def save_quote(quote_id: int, body: QuoteDraft, principal=Depends(require_user)):
    """Save a draft. Sending `dealId` links the quote to its deal, once; leaving it out
    or null leaves the link as it is."""
    gate("sales_quote.price", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    link = body.dealId if body.dealId is not None else builder.UNSET
    with http_errors(), get_conn() as c:
        return money_json(_with_checks(builder.save(
            c, quote_id, draft=_draft(body), actor=_subject(principal), deal_id=link,
            revision=body.revision)))


@router.post("/spendiq/quotes/{quote_id}/supersede")
def supersede(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.create", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(_with_checks(builder.supersede(c, quote_id,
                                                         actor=_subject(principal))))


# --- review -----------------------------------------------------------------

@router.post("/spendiq/quotes/{quote_id}/review")
def send_for_review(quote_id: int, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(_with_checks(builder.submit(c, quote_id, actor=_subject(principal))))


@router.post("/spendiq/quotes/{quote_id}/withdraw")
def withdraw(quote_id: int, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        quotes.withdraw(c, quote_id, actor=_subject(principal))
        return money_json(_with_checks(builder.get(c, quote_id)))


@router.post("/spendiq/quotes/{quote_id}/approve")
def approve(quote_id: int, body: Optional[ApproveBody] = None, principal=Depends(require_user)):
    """Approve, with the approver's conditions. Nobody approves their own quote."""
    gate("sales_quote.approve", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        quotes.approve(c, quote_id, approver=_subject(principal),
                       conditions=(body.conditions if body else None))
        return money_json(_with_checks(builder.get(c, quote_id)))


@router.post("/spendiq/quotes/{quote_id}/reject")
def reject(quote_id: int, body: RejectBody, principal=Depends(require_user)):
    """Send a quote back to draft with a reason. The same authority as approving it."""
    gate("sales_quote.approve", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        quotes.reject(c, quote_id, approver=_subject(principal), reason=body.reason)
        return money_json(_with_checks(builder.get(c, quote_id)))


# --- the drafting agent -----------------------------------------------------

class AgentTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    session_id: Optional[str] = None
    # The quote as the screen holds it now, unsaved edits included, so a proposal is
    # about what the seller is looking at. Left out, the saved quote is used.
    quote: Optional[Dict[str, Any]] = None


class DismissedProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(max_length=2000)
    message: Optional[str] = Field(default=None, max_length=quote_agent.MAX_MESSAGE)
    session_id: Optional[str] = None


@router.post("/spendiq/quotes/{quote_id}/agent")
def agent_turn(quote_id: int, body: AgentTurn, principal=Depends(require_user)):
    """One instruction, one proposal. Nothing on this path changes the quote: the
    proposal is applied, if the seller agrees, through PUT /spendiq/quotes/{id}."""
    gate("sales_quote.price", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        saved = builder.get(c, quote_id)
    # Whether it may be changed, and what it is tested against, are the saved quote's;
    # the lines are the screen's when it sent them.
    held = body.quote if isinstance(body.quote, dict) else {}
    quote = {**saved, **{k: held[k] for k in ("lines", "validUntil") if k in held}}
    try:
        with http_errors():
            return money_json(quote_agent.propose(
                quote, body.message, session_id=body.session_id,
                checks=pricing_checks.for_quote(saved.get("category"))))
    except quote_agent.AgentUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/spendiq/quotes/{quote_id}/agent/dismiss")
def dismiss_proposal(quote_id: int, body: DismissedProposal, principal=Depends(require_user)):
    """The seller turned a proposal down. Kept on the quote's trail: a proposal nobody
    wanted is what the agent should stop offering."""
    gate("sales_quote.price", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        saved = builder.get(c, quote_id)
        quotes.record_event(c, "sales_quote.proposal_dismissed",
                            {"sales_quote_id": quote_id, "quote_ref": saved["ref"]},
                            actor=_subject(principal), proposal=body.summary,
                            instruction=body.message, session_id=body.session_id)
        c.commit()
    return {"recorded": True}


# --- the selling entity -------------------------------------------------------

class EntityProfileBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    legalName: Optional[str] = None
    tradingName: Optional[str] = None
    logoText: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    web: Optional[str] = None
    companyNumber: Optional[str] = None
    vatNumber: Optional[str] = None
    footerNote: Optional[str] = None
    bank: Optional[Dict[str, Any]] = None


@router.get("/entity/profile")
def get_entity_profile(principal=Depends(require_user)):
    """Who every quote is from. A field nobody has filled in is null, never a stand-in."""
    with http_errors(), get_conn() as c:
        return money_json(entity_profile.load(c))


@router.put("/entity/profile")
def put_entity_profile(body: EntityProfileBody, principal=Depends(require_user)):
    gate("entity_profile.write", principal, agent=_AGENT, context={})
    with http_errors(), get_conn() as c:
        return money_json(entity_profile.save(c, body.model_dump(),
                                              actor=_subject(principal)))
