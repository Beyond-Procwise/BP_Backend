"""Sending a quote to its customer (QDA B12). Sending is the act that issues it.

One email: a cover with no figure in it, and the quote's PDF. It goes to a contact on the
quote's own account and to nobody else -- no caller supplies an address. It is sent by a
signed-in person; no agent sends.

The order is what makes it safe to retry:

  1. An approved quote is issued and numbered, and everything the send will need is checked
     before that is kept. A quote that could not be sent is not issued, and its number goes
     back.
  2. The issue is kept. Only then does the email go.
  3. That it went is recorded, with the message id and the hash of the PDF.

So an email that fails to leave finds the quote issued, numbered and not sent, and the same
call sends it. The other order -- email first, keep the issue second -- could put a number
in a customer's hands that the next quote is then given. The quote's row is locked from
the last check until the record is written, so two sends at once cannot both go.

No model is called. The cover is fixed words around the quote's number.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.services import agent_actions, email_dispatch_guard
from src.services.draft_assurance import payment_details
from src.services.rga import quote_pipeline
from src.services.rga.render import RenderedArtefact
from src.services.sell_side import quote_render, quotes
from src.services.sell_side._db import StateConflict, dict_cursor

SENT_EVENT = "sales_quote.sent"
BLOCKED_EVENT = "sales_quote.send_blocked"
FAILED_EVENT = "sales_quote.send_failed"
_AGENT = "SellSide"

# The one thing the cover cites. Its value is the quote's number, which exists once the
# quote is issued; until then the cover is its template.
REFERENCE_ID = "q1"
_REF = "{{ref:" + REFERENCE_ID + "}}"
SUBJECT_TEMPLATE = f"Quote {_REF}"
SENDABLE = ("approved", "issued")
# What the mail relay is given per attachment elsewhere in this product.
ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
POLICY_NOTE = "A quote is sent by a signed-in person. No agent sends one."

_ADDRESS = re.compile(r"[^@\s<>,;:\"()\[\]\\]+@[^@\s<>,;:\"()\[\]\\]+\.[^@\s<>,;:\"()\[\]\\.]+")


class SendRefused(StateConflict):
    """The quote did not pass every check, so nothing was sent. Carries all of them."""

    def __init__(self, checks: List[Dict[str, Any]]) -> None:
        self.checks = checks
        failing = [c for c in checks if not c["passed"]]
        super().__init__("the quote was not sent: " + "; ".join(c["message"] for c in failing))


class SendFailed(RuntimeError):
    """Every check passed and the mail relay did not take the message."""


@dataclass
class Prepared:
    """One quote, ready to send or not, and why."""

    quote: Dict[str, Any]
    contact: Optional[Dict[str, Any]]
    to: Optional[str]
    body_template: str
    subject: Optional[str]            # None until the quote has its number
    body: Optional[str]
    filename: str
    pdf: Optional[RenderedArtefact]
    sent: Optional[Dict[str, Any]]
    checks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def can_send(self) -> bool:
        return all(c["passed"] for c in self.checks)


def _subject_of(principal: Any) -> Optional[str]:
    return (getattr(principal, "subject", None) or "").strip() or None


def valid_address(value: Any) -> bool:
    """One plain address. Anything a mail header could be bent with is not one."""
    return isinstance(value, str) and bool(_ADDRESS.fullmatch(value))


def body_template(quote: Dict[str, Any], contact: Optional[Dict[str, Any]]) -> str:
    """The cover. It names who it is to and whose quote it is, and cites the quote. It
    holds no price, total, date or term: those are in the attachment, which is checked."""
    name = ((contact or {}).get("contact_name") or "").strip()
    greeting = f"Dear {name}," if name else "Hello,"
    return f"{greeting}\n\nPlease find attached quote {_REF} for {quote['account_name']}."


def _filled(template: str, number: str) -> str:
    return template.replace(_REF, number)


def _as_html(text: str) -> str:
    return "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))


def _filename(number: Optional[str]) -> str:
    safe = re.sub(r"[^A-Za-z0-9-]", "", number or "")
    return f"Quote-{safe}.pdf" if safe else "Quote.pdf"


def sent_record(cur: Any, sales_quote_id: int) -> Optional[Dict[str, Any]]:
    """The record that this quote went, if it did."""
    cur.execute(
        "SELECT created_at, details FROM proc.bp_agent_actions "
        "WHERE doc_type = 'sales_quote' AND doc_pk = %s AND action_type = %s "
        "ORDER BY created_at DESC LIMIT 1", (str(sales_quote_id), SENT_EVENT))
    row = cur.fetchone()
    if row is None:
        return None
    details = row["details"] or {}
    return {"sent_at": row["created_at"], "email_id": details.get("email_id"),
            "to": details.get("to"), "sent_by": details.get("actor")}


def _contact(cur: Any, quote: Dict[str, Any], contact_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """The contact this quote goes to: the one asked for, else the quote's own -- and only
    ever one on the quote's account."""
    wanted = contact_id if contact_id is not None else quote.get("contact_id")
    if wanted is None:
        return None
    cur.execute("SELECT contact_id, contact_name, contact_email FROM proc.bp_account_contact "
                "WHERE contact_id = %s AND account_id = %s", (wanted, quote["account_id"]))
    row = cur.fetchone()
    return dict(row) if row else None


def _cover_problem(text: str, quote: Dict[str, Any], contact: Optional[Dict[str, Any]],
                   number: str) -> Optional[str]:
    """A figure on the cover that is not the quote's number. Names are taken out first: an
    account called 'Unit 4 Fixings' is not a figure."""
    rest = text
    for known in (number, quote.get("account_name"), (contact or {}).get("contact_name")):
        if known:
            rest = rest.replace(str(known), " ")
    found = re.search(r"\d[\d.,]*", rest)
    return found.group(0) if found else None


def prepare(conn: Any, sales_quote_id: int, *, principal: Any, sender: Optional[str],
            contact_id: Optional[int] = None, policy_engine: Any = None) -> Prepared:
    """Everything a send needs, and every check it must pass. Sends nothing, changes nothing."""
    quote = quotes.get_quote(conn, sales_quote_id)
    internal = quotes.get_quote_internal(conn, sales_quote_id)
    cur = dict_cursor(conn)
    contact = _contact(cur, quote, contact_id)
    sent = sent_record(cur, sales_quote_id)
    number = quote.get("quote_number") or ""
    template = body_template(quote, contact)
    subject, body = _filled(SUBJECT_TEMPLATE, number), _filled(template, number)
    to = (contact or {}).get("contact_email")
    to = to.strip() if isinstance(to, str) else to

    checks: List[Dict[str, Any]] = []

    def check(check_id: str, passed: bool, message: str) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "severity": "block",
                       "message": message})

    status = quote["status"]
    check("quote_approved", status in SENDABLE,
          "The quote is approved." if status in SENDABLE else
          f"The quote is {status}; only an approved quote is sent.")
    check("quote_in_date", quote["valid_until"] >= dt.date.today(),
          "The quote is still valid." if quote["valid_until"] >= dt.date.today() else
          f"The quote expired on {quote['valid_until']:%-d %B %Y}.")
    check("not_already_sent", sent is None,
          "The quote has not been sent before." if sent is None else
          "The quote has already been sent.")
    numbered = status != "issued" or bool(number)
    check("quote_numbered", numbered,
          "The quote has its number, or is given one when it is sent." if numbered else
          "The quote was issued without a number and cannot be sent.")
    asked_for_unknown = contact is None and (contact_id is not None or quote.get("contact_id"))
    check("has_recipient", valid_address(to),
          "The quote has a contact with an email address." if valid_address(to) else
          "That contact is not on this quote's account." if asked_for_unknown else
          "The quote has no contact to send to." if contact is None else
          "The quote's contact has no usable email address.")

    pdf, problem = None, None
    if status in SENDABLE:
        try:
            pdf = quote_pipeline.produce(quote, internal, fmt="pdf")
        except quote_render.InternalFieldLeak:
            problem = "The quote document shows a figure that is ours, not the customer's."
        except (quote_pipeline.QuoteBlocked, quote_render.NotCustomerReady) as exc:
            problem = f"The quote document did not pass its checks: {exc}"
    else:
        problem = "There is no quote document until the quote is approved."
    check("named_docs_attached", pdf is not None,
          "The quote document is attached and passed its checks." if pdf is not None else problem)
    fits = pdf is None or len(pdf.content) <= ATTACHMENT_MAX_BYTES
    check("attachments_within_limit", fits,
          "The attachment is within the size the mail relay accepts." if fits else
          "The quote document is larger than the mail relay accepts.")

    cover = f"{subject}\n{body}"
    internal_id = next((i for i in (quote.get("quote_ref"),) if i and i in cover), None)
    try:
        quote_render.assert_no_internal_names(cover)
    except quote_render.InternalFieldLeak:
        internal_id = internal_id or "an internal field name"
    check("no_internal_ids", internal_id is None,
          "The email carries no internal reference." if internal_id is None else
          "The email carries an internal reference.")
    stray = _cover_problem(cover, quote, contact, number)
    try:
        quote_render.assert_no_internal_values(cover, quote, internal)
    except quote_render.InternalFieldLeak:
        stray = stray or "a cost or margin"
    check("figures_traced", stray is None,
          "The email carries no figure except the quote's number." if stray is None else
          "The email carries a figure that is not the quote's number.")
    bank = payment_details.account_details(subject, body)
    check("no_bank_details", not bank,
          "The email carries no bank details." if not bank else
          "The email carries bank details, which no email may.")

    check("mailbox_can_send", valid_address(sender),
          "A sending address is set up." if valid_address(sender) else
          "No sending address is set up, so nothing can be sent.")
    actor = _subject_of(principal)
    check("sent_by_a_person", actor is not None,
          "A signed-in person is sending this." if actor else POLICY_NOTE)
    over = (email_dispatch_guard.check_daily_volume(
        conn=conn, principal=principal, policy_engine=policy_engine) if actor else None)
    check("within_daily_limit", over is None,
          "You are within the daily limit for sent email." if over is None else
          f"Not sent: {over.reason}.")

    return Prepared(quote=quote, contact=contact, to=to if valid_address(to) else None,
                    body_template=template, subject=subject if number else None,
                    body=body if number else None, filename=_filename(number), pdf=pdf,
                    sent=sent, checks=checks)


def draft(conn: Any, sales_quote_id: int, *, principal: Any, sender: Optional[str],
          contact_id: Optional[int] = None, policy_engine: Any = None) -> Dict[str, Any]:
    """The cover draft for one quote, built from the quote id alone, with its checks."""
    p = prepare(conn, sales_quote_id, principal=principal, sender=sender,
                contact_id=contact_id, policy_engine=policy_engine)
    conn.rollback()                                      # a read; leave nothing open
    q, document_id = p.quote, f"quote_doc_{sales_quote_id}"
    return {
        "kind": "quote", "counterparty_kind": "customer", "trigger_type": "quote",
        "quote": {"id": sales_quote_id, "status": q["status"],
                  "quote_number": q.get("quote_number")},
        # The customer, in the field a supplier draft uses for the supplier.
        "supplier": {"id": q["account_id"], "name": q["account_name"]},
        "contact": ({"contact_id": p.contact["contact_id"],
                     "contact_name": p.contact["contact_name"]} if p.contact else None),
        "recipients": {"to": [p.to] if p.to else []},
        "subject_template": SUBJECT_TEMPLATE, "body_template": p.body_template,
        "subject": p.subject, "body": p.body,
        "references": [{"id": REFERENCE_ID, "kind": "quote",
                        "rendered_value": q.get("quote_number")}],
        "attachments": [{"document_id": document_id, "added_by": "agent",
                         "reference_id": REFERENCE_ID, "filename": p.filename}],
        "required_attachments": [document_id],
        "commercial": {"requires_approval": False},
        "policy": {"agent_send_prohibited": True, "note": POLICY_NOTE},
        "checks": p.checks, "can_send": p.can_send, "sent": p.sent,
    }


def _note(event: str, quote_id: int, actor: Optional[str], **details: Any) -> None:
    """A refused or failed send, on its own connection so it outlives the rollback."""
    agent_actions.record_action(
        phase="sales_quote", action_type=event, agent=_AGENT, doc_type="sales_quote",
        doc_pk=str(quote_id), status="denied" if event == BLOCKED_EVENT else "error",
        summary=f"{event} by {actor or 'unknown'}",
        details={"actor": actor, "sales_quote_id": quote_id, **details})


def send(conn: Any, sales_quote_id: int, *, principal: Any, mailer: Callable[..., Any],
         sender: Optional[str], contact_id: Optional[int] = None,
         policy_engine: Any = None) -> Dict[str, Any]:
    """Send the quote to its contact, issuing it first if it is still only approved.

    ``mailer(subject=, body=, recipients=, sender=, attachments=)`` is the mail relay; it
    returns something with ``success`` and ``message_id``. Raises ``SendRefused`` when a
    check fails (nothing sent, nothing issued) and ``SendFailed`` when the relay does not
    take the message (issued, not sent; call again).
    """
    actor = _subject_of(principal)

    def ready() -> Prepared:
        p = prepare(conn, sales_quote_id, principal=principal, sender=sender,
                    contact_id=contact_id, policy_engine=policy_engine)
        if not p.can_send:
            raise SendRefused(p.checks)
        return p

    try:
        quotes.get_quote(conn, sales_quote_id)             # NotFound before anything else
        # Locked before its status is read, so of two sends at once the second waits here,
        # finds the quote issued, and goes on to find it sent.
        cur = dict_cursor(conn)
        cur.execute("SELECT status FROM proc.bp_sales_quote WHERE sales_quote_id = %s "
                    "FOR UPDATE", (sales_quote_id,))
        if cur.fetchone()["status"] == "approved":
            quotes.issue(conn, sales_quote_id, actor=actor, before_commit=ready)
        else:
            conn.rollback()

        cur = dict_cursor(conn)
        try:
            cur.execute("SELECT 1 FROM proc.bp_sales_quote WHERE sales_quote_id = %s "
                        "FOR UPDATE", (sales_quote_id,))
            p = ready()
            artefact = {"sha256": hashlib.sha256(p.pdf.content).hexdigest(),
                        "bytes": len(p.pdf.content), "filename": p.filename,
                        "pack_hash": p.pdf.pack_hash, "ast_hash": p.pdf.ast_hash,
                        "renderer": p.pdf.renderer, "renderer_version": p.pdf.renderer_version}
            # Recorded on its own connection before the relay is called: an email this
            # cannot record does not go, and one that goes counts against the daily limit.
            agent_actions.record_action_or_fail(
                phase="communicate", action_type="email.send", agent=_AGENT,
                doc_type="sales_quote", doc_pk=str(sales_quote_id), status="allowed",
                summary=f"quote {p.quote['quote_number']} to its customer",
                details={"sales_quote_id": sales_quote_id, "principal": actor,
                         "recipients": [p.to], "account_id": p.quote["account_id"],
                         "decision": "allow", "egress": "amazon_ses",
                         "artefact_sha256": artefact["sha256"]})
            result = mailer(subject=p.subject, body=_as_html(p.body), recipients=[p.to],
                            sender=sender, attachments=[(p.pdf.content, p.filename)])
            if not getattr(result, "success", False):
                raise SendFailed(f"quote {p.quote['quote_number']} was not accepted for "
                                 "delivery; it is issued and has not been sent")
            email_id = getattr(result, "message_id", None)
            quotes.record_event(
                conn, SENT_EVENT, p.quote, actor=actor, quote_number=p.quote["quote_number"],
                to=p.to, contact_id=p.contact["contact_id"], email_id=email_id,
                subject=p.subject, artefact=artefact)
        except Exception:
            conn.rollback()
            raise
        conn.commit()
    except SendRefused as exc:
        _note(BLOCKED_EVENT, sales_quote_id, actor,
              failing=[c["id"] for c in exc.checks if not c["passed"]])
        raise
    except SendFailed as exc:
        _note(FAILED_EVENT, sales_quote_id, actor, reason=str(exc))
        raise

    return {"status": "sent", "sales_quote_id": sales_quote_id,
            "quote_number": p.quote["quote_number"], "to": p.to, "email_id": email_id,
            "artefact": artefact, "quote": quotes.get_quote(conn, sales_quote_id)}
