"""The Quote builder's API (UI BACKEND_GAPS.md, "Quote builder").

The paths are the ones the screen was built against; the record behind them is the
sell-side quote, so review, approval, numbering, the customer page and sending are the
/sales ones and are not repeated here.

A quote read returns the customer's quote only. Its cost record -- per line the tag,
cost and markup, with the margin totals and the limits it is tested against -- is read
and written on its own route, /spendiq/quotes/{id}/internal (ruled 2026-10-14). Every
route still asks margin.read with a STATED permit, as GET /sales/quotes/{id}/internal
does: a drafter works the cost record. Who acted is the token's subject; no body has a
field for it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from src.services import brand_logos, entity_profile
from src.services.sell_side import builder, pricing_checks, quote_agent, quotes
from src.services.sell_side import templates as quote_templates
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
    # The quote's layout: its sections in page order, with their headings, the words of
    # any free-text section, and the logo on the letterhead. Left out, it is unchanged.
    sections: Optional[List[Dict[str, Any]]] = None
    dealId: Optional[str] = None
    # What the last read returned as `revision`. Sent back, it stops a save from
    # silently undoing somebody else's later save.
    revision: Optional[str] = None


class InternalBody(BaseModel):
    """{lines: {<line id>: {tag?, cost?, markupPct?}}, revision?}. A key left out of a
    line keeps its value; null clears a cost or markup."""
    model_config = ConfigDict(extra="forbid")

    lines: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    revision: Optional[str] = None


class ApproveBody(BaseModel):
    conditions: List[str] = Field(default_factory=list)


class RejectBody(BaseModel):
    reason: str


def _with_checks(quote: Dict[str, Any]) -> Dict[str, Any]:
    """The quote as the customer's: nothing on it that is ours (builder.customer_fields)."""
    return builder.customer_fields(quote)


def _internal(quote: Dict[str, Any]) -> Dict[str, Any]:
    return builder.internal_fields(quote, pricing_checks.for_quote(quote.get("category")))


def _draft(body: QuoteDraft) -> Dict[str, Any]:
    return body.model_dump(exclude={"dealId", "revision"})


# --- templates --------------------------------------------------------------

class TemplateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: Optional[str] = None
    status: str = "active"
    category: Optional[str] = None
    lines: List[Dict[str, Any]] = Field(default_factory=list)
    termFields: List[str] = Field(default_factory=list)
    assumptionsHtml: Optional[str] = None
    sections: Optional[List[Dict[str, Any]]] = None


class FromTemplate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    templateKey: str
    currency: str
    dealId: Optional[str] = None
    customer: Optional[Dict[str, Any]] = None


@router.get("/spendiq/quote-templates")
def list_quote_templates(include_retired: bool = False, principal=Depends(require_user)):
    """The templates a quote can be started from. They hold no price and no cost."""
    gate("quote.read", principal, agent=_AGENT, context={"scope": "quote-templates"})
    with http_errors(), get_conn() as c:
        return money_json(quote_templates.list_templates(c, include_retired=include_retired))


@router.get("/spendiq/quote-templates/{key}")
def get_quote_template(key: str, principal=Depends(require_user)):
    gate("quote.read", principal, agent=_AGENT, context={"template_key": key})
    with http_errors(), get_conn() as c:
        return money_json(quote_templates.get(c, key))


@router.put("/spendiq/quote-templates/{key}")
def put_quote_template(key: str, body: TemplateBody, principal=Depends(require_user)):
    """Add a template, reword one, or retire one (`status: retired`)."""
    gate("quote_template.write", principal, agent=_AGENT, context={"template_key": key})
    with http_errors(), get_conn() as c:
        return money_json(quote_templates.save(
            c, key, name=body.name, description=body.description, status=body.status,
            body=body.model_dump(exclude={"name", "description", "status"}),
            actor=_subject(principal)))


@router.post("/spendiq/quotes/from-template")
def create_from_template(body: FromTemplate, principal=Depends(require_user)):
    """Start a quote from a template: its lines, with no price on any of them. On a deal
    when `dealId` is given, on none yet when it is not."""
    gate("sales_quote.create", principal, agent=_AGENT,
         context={"deal_id": body.dealId, "template_key": body.templateKey})
    _require_margin_permit(principal, {"deal_id": body.dealId})
    with http_errors(), get_conn() as c:
        template = quote_templates.get(c, body.templateKey)
        if template["status"] != "active":
            raise HTTPException(status_code=409, detail=f"template {body.templateKey} is "
                                "retired; choose another")
        c.rollback()                 # a read; create opens its own transaction
        draft = quote_templates.draft_from(template, currency=body.currency,
                                           customer=body.customer)
        quote = builder.create(c, deal_id=body.dealId, draft=draft,
                               created_by=_subject(principal), template_key=body.templateKey)
        return money_json({**_with_checks(quote),
                           "missing": quote_templates.missing(template, quote)})


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


@router.get("/spendiq/quotes/{quote_id}/internal")
def get_quote_internal(quote_id: int, principal=Depends(require_user)):
    """INTERNAL: the quote's cost record, keyed by the line ids the quote read returns:
    {quoteId, revision, lines: {<id>: {tag, cost, markupPct, costSource}}, totals: {cost,
    margin, marginFraction, costedLines, uncostedLines}, checks}. Never a customer's."""
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        return money_json(_internal(builder.get(c, quote_id)))


@router.put("/spendiq/quotes/{quote_id}/internal")
def save_quote_internal(quote_id: int, body: InternalBody, principal=Depends(require_user)):
    """Write a draft's cost record. The customer's price does not move here. A save gives
    the quote's lines new ids, so the answer carries both records: `quote` (the customer's)
    and `internal`, each with the new ids."""
    gate("sales_quote.price", principal, agent=_AGENT, context={"sales_quote_id": quote_id})
    _require_margin_permit(principal, {"sales_quote_id": quote_id})
    with http_errors(), get_conn() as c:
        saved = builder.save_internal(c, quote_id, lines=body.lines,
                                      actor=_subject(principal), revision=body.revision)
        return money_json({"quote": _with_checks(saved), "internal": _internal(saved)})


@router.get("/settings/quote")
def quote_settings(principal=Depends(require_user)):
    """The Quote builder's settings: {key: {value, source, locked}}. A null value is unset,
    and a builder quote cannot go for review or be sent while the margin floor or its tax
    rate is unset. They are governed (bp_policy `sales_quote_pricing`), so `locked`."""
    return pricing_checks.settings()


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
        try:
            return money_json(_with_checks(builder.submit(c, quote_id,
                                                          actor=_subject(principal))))
        except builder.PolicyUnset as exc:   # before http_errors flattens it to text
            raise HTTPException(status_code=409, detail={
                "message": str(exc), "codes": exc.codes}) from None


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
        # A quote that was not started from a template is read against the plain layout,
        # so its sections can be filled and its sums worked out just the same.
        template = quote_templates.find(c, saved.get("templateKey")) or quote_templates.BLANK
    # Whether it may be changed, which template it follows and what it is tested against
    # are the saved quote's; what is on it is the screen's when it sent that.
    held = body.quote if isinstance(body.quote, dict) else {}
    quote = {**saved, **{k: held[k] for k in ("lines", "validUntil", "customer", "terms",
                                               "sections", "scopeHtml", "assumptionsHtml")
                         if k in held}}
    # The screen holds the cost record apart, so its lines may come without it: each line
    # it read gets its saved tag, cost and markup unless the screen sent its own.
    own = {l["id"]: l for l in saved.get("lines") or []}
    quote["lines"] = [
        {**{k: own[l["id"]].get(k) for k in builder.INTERNAL_LINE_KEYS if k not in l}, **l}
        if isinstance(l, dict) and l.get("id") in own else l
        for l in quote.get("lines") or []]
    try:
        with http_errors():
            return money_json(quote_agent.propose(
                quote, body.message, session_id=body.session_id, template=template,
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
    # the company's default logo: one of GET /entity/logos, or null for none
    logoId: Optional[int] = None


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


# --- logos --------------------------------------------------------------------
#
# Uploading and removing a logo are changes to the company profile, so they ask the
# same permit. Which logo is the default is a field of the profile (`logoId`); a quote
# or template swaps to another by naming it on its letterhead section.

@router.get("/entity/logos")
def list_logos(principal=Depends(require_user)):
    with http_errors(), get_conn() as c:
        return money_json(brand_logos.list_logos(c))


@router.post("/entity/logos")
def upload_logo(file: UploadFile = File(...), name: Optional[str] = Form(default=None),
                principal=Depends(require_user)):
    """A PNG or JPEG, up to 1 MB. It joins the list; it is not made the default."""
    gate("entity_profile.write", principal, agent=_AGENT, context={"logo": "upload"})
    data = file.file.read(brand_logos.MAX_BYTES + 1)
    with http_errors(), get_conn() as c:
        return money_json(brand_logos.add(c, name=name or file.filename, data=data,
                                          actor=_subject(principal)))


@router.get("/entity/logos/{logo_id}/image")
def logo_image(logo_id: int, principal=Depends(require_user)):
    with http_errors(), get_conn() as c:
        data, content_type = brand_logos.image(c, logo_id)
    return Response(content=data, media_type=content_type,
                    headers={"X-Content-Type-Options": "nosniff",
                             "Cache-Control": "private, max-age=3600"})


@router.delete("/entity/logos/{logo_id}")
def remove_logo(logo_id: int, principal=Depends(require_user)):
    gate("entity_profile.write", principal, agent=_AGENT, context={"logo": logo_id})
    with http_errors(), get_conn() as c:
        default = entity_profile.load(c).get("logoId")
        c.rollback()
        brand_logos.remove(c, logo_id, default_logo_id=default)
    return {"removed": True}
