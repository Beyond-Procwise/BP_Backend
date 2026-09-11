"""Sell-side API: accounts, opportunities, outbound quotes, outcomes.

Who acted -- created_by, approver, recorded_by -- is the token's subject and
nothing else; no body here has a field for it. Approving a quote is a transact
action and issuing one is communicate, so both need a stated permit
(deploy/sql/2026-09-11_reseller_governance.sql). The customer endpoints go
through quote_render, which is the control on cost and margin.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Not src.services.db.get_conn: that opens autocommit connections, under which the
# sell-side services' rollback-before-raise is a no-op and FOR UPDATE locks don't hold.
from src.services.sell_side._db import transactional_conn as get_conn
from src.services.sell_side import (accounts, calibration, opportunities, outcomes,
                                    quote_render, quotes)

from api.auth import require_user
from api.endpoint_gate import require as gate
from api.sell_side_http import http_errors, money_json

router = APIRouter(prefix="/sales", tags=["Sales"])
_AGENT = "SalesRouter"
MARGIN_NOTE = ("This is front-end margin only: distributor back-end rebates are not "
               "modelled, so it understates what a line may finally earn.")


def _subject(principal) -> Optional[str]:
    return getattr(principal, "subject", None) or None


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


class QuoteLine(BaseModel):
    catalog_item_id: int
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
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
def get_account(account_id: str):
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


# --- opportunities ----------------------------------------------------------

@router.post("/opportunities")
def create_opportunity(body: OpportunityBody, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    with http_errors(), get_conn() as c:
        return money_json(opportunities.create_opportunity(c, **body.model_dump()))


@router.get("/opportunities")
def list_opportunities(account_id: Optional[str] = None, outcome: Optional[str] = None,
                       limit: int = 100):
    with get_conn() as c:
        rows = opportunities.list_opportunities(c, account_id=account_id, outcome=outcome, limit=limit)
    return money_json({"count": len(rows), "opportunities": rows})


@router.get("/opportunities/{opportunity_id}")
def get_opportunity(opportunity_id: int):
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
    gate("sales.write", principal, agent=_AGENT, context={"account_id": body.account_id})
    data = body.model_dump()
    data["lines"] = [l.model_dump() for l in body.lines]
    with http_errors(), get_conn() as c:
        return money_json(quotes.create_draft(c, created_by=_subject(principal), **data))


@router.get("/quotes/{quote_id}")
def get_quote(quote_id: int):
    """INTERNAL view: carries cost and margin. Never hand this to a customer."""
    with http_errors(), get_conn() as c:
        return money_json({**quotes.get_quote(c, quote_id), "margin_note": MARGIN_NOTE})


@router.post("/quotes/{quote_id}/submit")
def submit(quote_id: int, principal=Depends(require_user)):
    gate("sales.write", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quotes.submit(c, quote_id, actor=_subject(principal)))


@router.post("/quotes/{quote_id}/approve")
def approve(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.approve", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quotes.approve(c, quote_id, approver=_subject(principal)))


@router.post("/quotes/{quote_id}/issue")
def issue(quote_id: int, principal=Depends(require_user)):
    gate("sales_quote.issue", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(quotes.issue(c, quote_id, actor=_subject(principal)))


@router.get("/quotes/{quote_id}/customer")
def customer_view(quote_id: int):
    with http_errors(), get_conn() as c:
        return money_json(quote_render.customer_view(quotes.get_quote(c, quote_id)))


@router.get("/quotes/{quote_id}/customer.html", response_class=HTMLResponse)
def customer_html(quote_id: int):
    with http_errors(), get_conn() as c:
        return HTMLResponse(quote_render.render_html(
            quote_render.customer_view(quotes.get_quote(c, quote_id))))


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
