"""Sell-side API: accounts, opportunities, outbound quotes, outcomes.

Who acted -- created_by, approver, recorded_by -- is the token's subject and
nothing else; no body here has a field for it. Approving a quote is a transact
action and issuing one is communicate, so both need a stated permit
(deploy/sql/2026-09-11_reseller_governance.sql). The customer endpoints are the
control on cost and margin: the JSON view goes through quote_render, and the page
and its PDF through rga.quote_pipeline, which returns nothing that failed its gate.

A quote is two reads behind two permissions. GET /quotes/{id} is the customer
record and has no cost or margin in it; GET /quotes/{id}/internal is what the quote
costs us. Every read that returns cost or margin -- that internal read and both
opportunity reads -- asks margin.read and must be permitted by a stated policy
(deploy/sql/2026-09-14_sales_margin_read_authority.sql). require_user alone is
not a control here: with ASK_AUTH_MODE=off it yields no principal rather than
refusing.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

# Not src.services.db.get_conn: that opens autocommit connections, under which the
# sell-side services' rollback-before-raise is a no-op and FOR UPDATE locks don't hold.
from src.services.sell_side._db import transactional_conn as get_conn
from src.services.sell_side import (accounts, agreed_pricing, special_bids, calibration, opportunities, outcomes,
                                    quote_render, quote_send, quotes)

from src.services import agent_actions
from src.services.rga import quote_pipeline
from src.services.rga.render.html import PAGE_CSP  # one policy, header and page

from api.auth import require_user
from api.endpoint_gate import NotPermitted, require as gate
from api.sell_side_http import http_errors, money_json

router = APIRouter(prefix="/sales", tags=["Sales"])
_AGENT = "SalesRouter"
MARGIN_NOTE = ("This is front-end margin only: distributor back-end rebates are not "
               "modelled, so it understates what a line may finally earn.")


def _subject(principal) -> Optional[str]:
    return getattr(principal, "subject", None) or None


def _require_margin_permit(principal, context: dict) -> None:
    """Cost and margin go only to a caller a STATED policy permits.

    margin.read is a read, and a read no policy speaks to is allowed for every
    role -- the anonymous caller included. An allow from that default carries no
    policy_id, so it is refused here: a missing SalesMarginReadAuthorityPolicy row
    withholds margin from everyone rather than handing it to anyone.
    """
    decision = gate("margin.read", principal, agent=_AGENT, context=context)
    if decision.policy_id:
        return
    agent_actions.record_action_or_fail(
        phase="authorize", action_type="margin.read", agent=_AGENT, status="denied",
        summary="no stated permit for margin.read; the reversible-read default does "
                "not extend to cost and margin",
        details={**context, "principal": _subject(principal),
                 "allowed_by": decision.policy_name})
    raise NotPermitted("margin.read needs a stated permit and none is active, so "
                       "cost and margin are withheld.")


class AccountBody(BaseModel):
    account_name: str
    account_id: Optional[str] = None
    trading_name: Optional[str] = None
    also_supplier_id: Optional[str] = None
    registration_number: Optional[str] = None
    vat_number: Optional[str] = None
    country: Optional[str] = None
    default_currency: Optional[str] = None
    payment_terms: Optional[str] = None
    credit_limit_amount: Optional[Decimal] = None
    account_owner_email: Optional[str] = None


class ContactBody(BaseModel):
    contact_name: str
    contact_role: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    is_primary: bool = False


class ScopeBody(BaseModel):
    completeness: str
    covers_from: Optional[dt.date] = None
    covers_to: Optional[dt.date] = None
    note: Optional[str] = None


class OpportunityBody(BaseModel):
    account_id: str
    opportunity_type: str
    catalog_item_id: Optional[int] = None
    currency: Optional[str] = None
    expected_quantity: Optional[Decimal] = None
    expected_unit_price: Optional[Decimal] = None
    phase_id: Optional[str] = "sales.opportunity"
    subprocess_id: Optional[str] = "sales.opportunity.qualified"
    detector_type: Optional[str] = None
    reason_codes: Optional[List[str]] = None


class JustificationBody(BaseModel):
    kind: str
    claim: str
    evidence_ref: Optional[str] = None
    evidence_value: Optional[Decimal] = None
    customer_safe: bool = True


class StageBody(BaseModel):
    phase_id: str
    subprocess_id: Optional[str] = None


class AgreedPriceBody(BaseModel):
    distributor_id: str
    distributor_sku: str
    price_kind: str                      # fixed | discount_pct
    currency: str
    valid_from: dt.date
    unit_price: Optional[Decimal] = None
    discount_pct: Optional[Decimal] = None
    min_quantity: Decimal = Field(default=Decimal(0), ge=0)
    valid_to: Optional[dt.date] = None
    reference: Optional[str] = None


class SpecialBidBody(BaseModel):
    account_id: str
    distributor_id: str
    distributor_sku: str
    bid_reference: str
    cost_price: Decimal = Field(ge=0)
    currency: str
    valid_from: dt.date
    valid_to: Optional[dt.date] = None   # required, but refused by the service with a reason
    min_quantity: Decimal = Field(default=Decimal(0), ge=0)
    max_quantity: Optional[Decimal] = None


class QuoteLine(BaseModel):
    catalog_item_id: int
    quantity: Decimal = Field(gt=0)
    # left out, the line takes the customer's agreed price; with none, the quote is refused
    unit_price: Optional[Decimal] = Field(default=None, ge=0)
    sales_opportunity_id: Optional[int] = None
    justification_id: Optional[int] = None


class QuoteBody(BaseModel):
    account_id: str
    currency: str
    valid_until: dt.date
    lines: List[QuoteLine]
    contact_id: Optional[int] = None
    quote_date: Optional[dt.date] = None
    supersedes_id: Optional[int] = None


class SendBody(BaseModel):
    """Which contact the quote goes to, if not the quote's own. Never an address."""
    contact_id: Optional[int] = None


class OutcomeBody(BaseModel):
    outcome: str
    outcome_date: dt.date
    lost_reason: Optional[str] = None
    competitor_name: Optional[str] = None


# --- accounts ---------------------------------------------------------------

@router.post("/accounts")
def create_account(body: AccountBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    fields = body.model_dump(exclude_none=True)
    name = fields.pop("account_name")
    with http_errors(), get_conn() as c:
        return money_json(accounts.create_account(c, account_name=name, **fields))


@router.get("/accounts/{account_id}")
def get_account(account_id: str, principal=Depends(require_user)):
    gate("account.read", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return money_json(accounts.get_account(c, account_id))


@router.post("/accounts/{account_id}/contacts")
def add_contact(account_id: str, body: ContactBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return money_json(accounts.add_contact(c, account_id, **body.model_dump()))


@router.put("/accounts/{account_id}/history-scope/{source_kind}")
def set_scope(account_id: str, source_kind: str, body: ScopeBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return money_json(accounts.set_history_scope(c, account_id, source_kind=source_kind, **body.model_dump()))


# --- agreed prices and special bids -------------------------------------------

@router.post("/accounts/{account_id}/prices")
def register_agreed_price(account_id: str, body: AgreedPriceBody, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return money_json(agreed_pricing.register(
            c, account_id=account_id, created_by=_subject(principal), **body.model_dump()))


@router.get("/accounts/{account_id}/prices")
def list_agreed_prices(account_id: str, principal=Depends(require_user)):
    gate("account.read", principal, agent=_AGENT, context={"account_id": account_id})
    with http_errors(), get_conn() as c:
        return money_json({"prices": agreed_pricing.list_for_account(c, account_id)})


@router.post("/prices/{account_price_id}/withdraw")
def withdraw_agreed_price(account_price_id: int, principal=Depends(require_user)):
    gate("account.write", principal, agent=_AGENT, context={"account_price_id": account_price_id})
    with http_errors(), get_conn() as c:
        return money_json(agreed_pricing.withdraw(c, account_price_id, actor=_subject(principal)))


# A special bid is our COST for one customer's deal, so reading or changing one needs the
# same stated margin permit as reading a quote's cost.

@router.post("/special-bids")
def register_special_bid(body: SpecialBidBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    _require_margin_permit(principal, {"account_id": body.account_id})
    with http_errors(), get_conn() as c:
        return money_json(special_bids.register(
            c, created_by=_subject(principal), **body.model_dump()))


@router.get("/accounts/{account_id}/special-bids")
def list_special_bids(account_id: str, principal=Depends(require_user)):
    _require_margin_permit(principal, {"account_id": account_id})
    with http_errors(), get_conn() as c:
        return money_json({"special_bids": special_bids.list_for_account(c, account_id)})


@router.post("/special-bids/{special_bid_id}/withdraw")
def withdraw_special_bid(special_bid_id: int, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"special_bid_id": special_bid_id})
    _require_margin_permit(principal, {"special_bid_id": special_bid_id})
    with http_errors(), get_conn() as c:
        return money_json(special_bids.withdraw(c, special_bid_id, actor=_subject(principal)))


# --- opportunities ----------------------------------------------------------

@router.post("/opportunities")
def create_opportunity(body: OpportunityBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    with http_errors(), get_conn() as c:
        return money_json(opportunities.create_opportunity(c, **body.model_dump()))


@router.get("/opportunities")
def list_opportunities(account_id: Optional[str] = None, outcome: Optional[str] = None,
                       limit: int = 100, principal=Depends(require_user)):
    _require_margin_permit(principal, {"account_id": account_id})
    with get_conn() as c:
        rows = opportunities.list_opportunities(c, account_id=account_id, outcome=outcome, limit=limit)
    return money_json({"count": len(rows), "opportunities": rows})


@router.get("/opportunities/{opportunity_id}")
def get_opportunity(opportunity_id: int, principal=Depends(require_user)):
    _require_margin_permit(principal, {"sales_opportunity_id": opportunity_id})
    with http_errors(), get_conn() as c:
        return money_json(opportunities.get_opportunity(c, opportunity_id))


@router.post("/opportunities/{opportunity_id}/justifications")
def add_justification(opportunity_id: int, body: JustificationBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_opportunity_id": opportunity_id})
    with http_errors(), get_conn() as c:
        return money_json(opportunities.add_justification(c, opportunity_id, **body.model_dump()))


@router.put("/opportunities/{opportunity_id}/stage")
def set_stage(opportunity_id: int, body: StageBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_opportunity_id": opportunity_id})
    with http_errors(), get_conn() as c:
        return money_json(opportunities.set_stage(c, opportunity_id, **body.model_dump()))


# --- quotes -----------------------------------------------------------------

@router.post("/quotes")
def create_quote(body: QuoteBody, principal=Depends(require_user)):
    gate("sales_quote.create", principal, agent=_AGENT, context={"account_id": body.account_id})
    data = body.model_dump()
    data["lines"] = [l.model_dump() for l in body.lines]
    with http_errors(), get_conn() as c:
        return money_json(quotes.create_draft(c, created_by=_subject(principal), **data))


@router.get("/quotes/{quote_id}")
def get_quote(quote_id: int, principal=Depends(require_user)):
    """The quote without what it costs us. Cost and margin are /internal."""
    gate("quote.read", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quotes.get_quote(c, quote_id))


@router.get("/quotes/{quote_id}/internal")
def get_quote_internal(quote_id: int, principal=Depends(require_user)):
    """INTERNAL: cost and margin per quote and per line id. Never hand this to a customer."""
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json({**quotes.get_quote_internal(c, quote_id),
                           "margin_note": MARGIN_NOTE})


@router.post("/quotes/{quote_id}/submit")
def submit(quote_id: int, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quotes.submit(c, quote_id, actor=_subject(principal)))


def floor_override(principal, quote_id: int):
    """What quotes.approve asks when a builder quote misses the margin floor: may THIS
    approver override it? Only a stated permit for sales_quote.override_margin_floor says
    yes; the ask is on the audit spine either way."""
    def _may(block: dict) -> bool:
        try:
            gate("sales_quote.override_margin_floor", principal, agent=_AGENT,
                 context={"sales_quote_id": quote_id, "code": block["code"],
                          "floor_pct": block["floorPct"]})
        except NotPermitted:
            return False
        return True
    return _may


def margin_floor_refused(exc: "quotes.MarginBelowFloor") -> HTTPException:
    return HTTPException(status_code=409, detail={
        "message": str(exc), "codes": exc.codes,
        "floorPct": exc.block["floorPct"]})


@router.post("/quotes/{quote_id}/approve")
def approve(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.approve", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        try:
            return money_json(quotes.approve(c, quote_id, approver=_subject(principal),
                                             may_override_floor=floor_override(principal, quote_id)))
        except quotes.MarginBelowFloor as exc:   # before http_errors flattens it to text
            raise margin_floor_refused(exc) from None


@router.post("/quotes/{quote_id}/issue")
def issue(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.issue", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quotes.issue(c, quote_id, actor=_subject(principal)))


@router.get("/quotes/{quote_id}/customer")
def customer_view(quote_id: int, principal=Depends(require_user)):
    gate("quote.read", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quote_render.customer_view(
            quotes.get_quote(c, quote_id), quotes.get_quote_internal(c, quote_id)))


@router.get("/quotes/{quote_id}/customer.html", response_class=HTMLResponse)
def customer_html(quote_id: int, principal=Depends(require_user)):
    gate("quote.read", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        page = quote_pipeline.produce(quotes.get_quote(c, quote_id),
                                      quotes.get_quote_internal(c, quote_id))
        return HTMLResponse(page.content.decode("utf-8"),
                            headers={"Content-Security-Policy": PAGE_CSP})


@router.get("/quotes/{quote_id}/customer.pdf")
def customer_pdf(quote_id: int, principal=Depends(require_user)):
    """The same page, printed. One document, checked once, two files."""
    gate("quote.read", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        pdf = quote_pipeline.produce(quotes.get_quote(c, quote_id),
                                     quotes.get_quote_internal(c, quote_id), fmt="pdf")
        return Response(pdf.content, media_type=pdf.media_type, headers={
            "Content-Disposition": f'inline; filename="quote-{quote_id}.pdf"'})


def _sender(request: Request) -> Optional[str]:
    settings = getattr(getattr(request.app.state, "agent_nick", None), "settings", None)
    return (str(getattr(settings, "ses_default_sender", "") or "").strip() or None)


@router.get("/quotes/{quote_id}/send-draft")
def send_draft(quote_id: int, request: Request, contact_id: Optional[int] = None,
               principal=Depends(require_user)):
    """The cover email for this quote and whether it can be sent. Sends nothing."""
    gate("quote.read", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quote_send.draft(c, quote_id, principal=principal,
                                           sender=_sender(request), contact_id=contact_id))


@router.post("/quotes/{quote_id}/send")
def send(quote_id: int, request: Request, body: Optional[SendBody] = None,
         principal=Depends(require_user)):
    """Send the quote to its customer. Sending an approved quote is what issues it, so it
    asks for both permits; neither is granted by default."""
    context = {"sales_quote_id": quote_id}
    gate("sales_quote.send", principal, agent=_AGENT, context=context)
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if agent_nick is None:
        raise HTTPException(status_code=503, detail="the mail relay is not available")
    from src.services.email_service import EmailService

    with http_errors(), get_conn() as c:
        if quotes.get_quote(c, quote_id)["status"] == "approved":
            gate("sales_quote.issue", principal, agent=_AGENT, context=context)
        try:
            return money_json(quote_send.send(
                c, quote_id, principal=principal, sender=_sender(request),
                mailer=EmailService(agent_nick).send_email,
                contact_id=body.contact_id if body else None))
        except quote_send.SendRefused as exc:      # before http_errors flattens it to text
            raise HTTPException(status_code=409, detail={
                "message": str(exc), "checks": exc.checks}) from None
        except quote_send.SendFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None


@router.post("/quotes/{quote_id}/outcome")
def record_outcome(quote_id: int, body: OutcomeBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(outcomes.record_outcome(c, sales_quote_id=quote_id,
                                                   recorded_by=_subject(principal),
                                                   **body.model_dump()))


@router.post("/calibrate")
def calibrate(principal=Depends(require_user)):
    gate("sales.calibrate", principal, agent=_AGENT)
    with get_conn() as c:
        return money_json({"calibrations": [c_.__dict__ for c_ in calibration.calibrate(c)]})
