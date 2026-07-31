"""Query it — one click from "we found this" to "we asked the supplier about it".

The distance between finding money and getting it back is usually an email nobody writes.
This closes it: a draft built from the finding, a human review, a send, and a timestamp on
the row so the same supplier is never chased twice for the same invoice.

Grounding is a property of the construction, not a check afterwards. There is NO model in
this path. The template is a Python format string and every figure slot is filled from the
stored discrepancy row — the same strings the drawer shows — so the email cannot state a
number the database does not hold. That matters more here than anywhere else in the
product: this message accuses a named supplier of over-billing.

Only discrepancy findings are queryable. An opportunity is our own analysis of a price;
there is nothing to put to a supplier.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from src.services import agent_actions
from src.services.value_summary_service import DISCREPANCY_VALUE_TYPES, parse_amount

logger = logging.getLogger(__name__)

AGENT = "value_found_query"

# The governed override lives in proc.bp_prompt under this name; absent means the built-in
# template below is used verbatim.
PROMPT_NAME = "value_found_query_email"


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------

# One body per kind of claim, because "you billed more than the PO" and "you billed us
# twice" ask the supplier for different things. Every {slot} is filled from the row.
DEFAULT_TEMPLATE = {
    "subject": "Query on {doc_ref} against {po_ref}",
    "body": (
        "Hello {supplier_name},\n\n"
        "We are reviewing invoice {doc_ref} against purchase order {po_ref} and it appears "
        "to be {delta} above the ordered value.\n\n"
        "Could you confirm whether this difference is expected? If it is not, please issue "
        "a credit note for {delta}. If it is, a short explanation of what the additional "
        "charge covers would let us close this off.\n\n"
        "Many thanks,\n"
        "Accounts Payable"
    ),
}

DUPLICATE_TEMPLATE = {
    "subject": "Query on {doc_ref} — possible duplicate of {duplicate_of}",
    "body": (
        "Hello {supplier_name},\n\n"
        "Invoice {doc_ref} appears to duplicate invoice {duplicate_of}: both are for "
        "{delta} against purchase order {po_ref}.\n\n"
        "Could you confirm whether {doc_ref} was issued in error? If it was, and it has "
        "already been settled, please issue a credit note for {delta}. If the two invoices "
        "are for genuinely different work, an explanation of what each covers would let us "
        "close this off.\n\n"
        "Many thanks,\n"
        "Accounts Payable"
    ),
}


def _governed_template() -> Optional[dict]:
    """The DB-governed override, if one is published. Fail-open: governance being
    unavailable must never stop a buyer querying an invoice."""
    try:
        from src.services.governance_tools import tools as governance
        row = governance.get_prompt(PROMPT_NAME) or {}
        subject = (row.get("prompt_text_subject") or row.get("subject") or "").strip()
        body = (row.get("prompt_text") or row.get("body") or "").strip()
        if body:
            return {"subject": subject or DEFAULT_TEMPLATE["subject"], "body": body}
    except Exception:
        logger.debug("governed query template unavailable; using the built-in", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# supplier_email comes from the SUPPLIER ID on the invoice, not from matching the supplier's
# name: names drift ("Techworld" / "Techworld Ltd.") and an email sent to the wrong company
# about their competitor's invoice is not a formatting slip.
_LOAD_SQL = """
    SELECT e.discrepancy_id, e.issue_type, e.status, e.doc_type, e.doc_pk_candidate,
           e.raw_value, e.expected_value, e.computed_value, e.notes, e.query_sent_at,
           i.po_id, i.currency, NULLIF(i.deal_id, '') AS deal_id, i.supplier_id,
           s.supplier_name, s.contact_email_1 AS supplier_email
      FROM proc.bp_extraction_discrepancy e
      LEFT JOIN proc.bp_invoice_trgt i
             ON e.doc_type = 'invoice' AND i.invoice_id = e.doc_pk_candidate
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id
     WHERE e.discrepancy_id = %s
"""


def _discrepancy_id(finding_id: str) -> int:
    """`disc:<id>` -> id. Task 2 owns this format; anything else is not queryable."""
    text = str(finding_id or "")
    if not text.startswith("disc:"):
        raise ValueError(f"only discrepancy findings can be queried, not {finding_id!r}")
    try:
        return int(text.split(":", 1)[1])
    except (IndexError, ValueError):
        raise ValueError(f"malformed finding id {finding_id!r}") from None


def _load(cur, discrepancy_id: int) -> dict:
    cur.execute(_LOAD_SQL, (discrepancy_id,))
    cols = [d[0] for d in (cur.description or [])]
    rows = cur.fetchall()
    if not rows:
        raise ValueError(f"finding disc:{discrepancy_id} not found")
    return dict(zip(cols, rows[0]))


def _require_queryable(row: dict) -> None:
    if row.get("issue_type") not in DISCREPANCY_VALUE_TYPES:
        raise ValueError(f"disc:{row.get('discrepancy_id')} is not a value finding "
                         f"({row.get('issue_type')})")
    if row.get("status") != "open":
        raise ValueError(f"disc:{row.get('discrepancy_id')} is not open "
                         f"(status {row.get('status')}) — nothing to query")


# ---------------------------------------------------------------------------
# The draft
# ---------------------------------------------------------------------------

def _duplicate_of(notes: Any) -> Optional[str]:
    """The other invoice named in a duplicate finding's notes. The detector writes
    'possible duplicate of <id> (date): ...', so the reference is read back rather than
    re-derived — one place decides which invoice this duplicates."""
    text = str(notes or "")
    marker = "possible duplicate of "
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1].strip()
    return tail.split(" ", 1)[0].strip(":,") or None


def figures(row: dict) -> dict:
    """Every number and reference the email may contain, formatted once, from stored values.

    The delta is rendered with the same two-decimal form the drawer uses, so a buyer reading
    the finding and the supplier reading the email see the same figure character for
    character.
    """
    delta = parse_amount(row.get("computed_value"))
    if delta is None:
        observed, expected = parse_amount(row.get("raw_value")), parse_amount(row.get("expected_value"))
        delta = abs(observed - expected) if observed is not None and expected is not None else None
    if delta is None:
        raise ValueError(f"disc:{row.get('discrepancy_id')} has no amount to query")
    # The currency is named. The corpus bills in five of them, and "please issue a credit
    # note for 1,321.06" to a supplier who invoices in INR is a different demand from the
    # one intended. Where the document never stated a currency, the amount goes out bare
    # rather than wearing a guessed symbol.
    currency = str(row.get("currency") or "").strip().upper()
    amount = f"{abs(delta):,.2f}"
    out = {
        "delta": f"{amount} {currency}".strip(),
        "amount": amount,
        "currency": currency,
        "doc_ref": str(row.get("doc_pk_candidate") or ""),
        "po_ref": str(row.get("po_id") or "the purchase order"),
    }
    dup = _duplicate_of(row.get("notes"))
    if dup:
        out["duplicate_of"] = dup
    return out


def build_draft(finding_id: str, conn=None) -> dict:
    """The email a human is about to review. Never sends, never stamps anything.

    A supplier with no email on file still gets a draft — it is useful to a buyer who will
    look the address up — but `to` is None and the send endpoint refuses it.
    """
    if conn is not None:
        return _build_draft(conn, finding_id)
    from src.services.db import get_conn
    with get_conn() as own:
        return _build_draft(own, finding_id)


def _build_draft(conn, finding_id: str) -> dict:
    row = _load(conn.cursor(), _discrepancy_id(finding_id))
    _require_queryable(row)
    fig = figures(row)
    slots = {**fig, "supplier_name": row.get("supplier_name") or "Supplier"}
    template = (_governed_template()
                or (DUPLICATE_TEMPLATE if row.get("issue_type") == "duplicate_invoice"
                                          and "duplicate_of" in fig
                    else DEFAULT_TEMPLATE))
    return {
        "finding_id": f"disc:{row['discrepancy_id']}",
        "to": (row.get("supplier_email") or None),
        "subject": _fill(template["subject"], slots),
        "body": _fill(template["body"], slots),
        "figures": fig,
        "supplier_name": row.get("supplier_name"),
        "query_sent_at": _iso(row.get("query_sent_at")),
    }


def _fill(template: str, slots: dict) -> str:
    """format_map with a forgiving mapping: a governed template naming a slot this finding
    does not have leaves the placeholder visible for a human to notice, rather than raising
    and blocking the query outright."""
    class _Slots(dict):
        def __missing__(self, key):  # noqa: D105
            return "{" + key + "}"
    return str(template).format_map(_Slots(slots))


def _iso(value) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


# ---------------------------------------------------------------------------
# The send
# ---------------------------------------------------------------------------

def _send_email(*, to: str, subject: str, body: str, agent_nick) -> bool:
    """The SES boundary, isolated so the send path is testable without a mail server."""
    from src.services.email_service import EmailService
    sender = getattr(getattr(agent_nick, "settings", None), "ses_default_sender", None) \
        or "noreply@procwise.co.uk"
    result = EmailService(agent_nick).send_email(subject=subject, body=body,
                                                 recipients=to, sender=sender)
    return bool(getattr(result, "success", result))


_STAMP_SQL = """
    UPDATE proc.bp_extraction_discrepancy
       SET query_sent_at = %s
     WHERE discrepancy_id = %s
"""


def send_query(finding_id: str, *, to: str, subject: str, body: str, agent_nick,
               conn=None) -> dict:
    """Send the reviewed draft and record that it went.

    The row is stamped ONLY after SES accepts the message. A failed send that still stamped
    would make an unasked question look asked, and the finding would sit there waiting for a
    reply nobody was ever asked for.
    """
    if conn is not None:
        return _send(conn, finding_id, to, subject, body, agent_nick)
    from src.services.db import get_conn
    with get_conn() as own:
        return _send(own, finding_id, to, subject, body, agent_nick)


def _send(conn, finding_id: str, to: str, subject: str, body: str, agent_nick) -> dict:
    if not str(to or "").strip():
        raise ValueError("a query needs a recipient — no supplier email on file")
    cur = conn.cursor()
    row = _load(cur, _discrepancy_id(finding_id))
    # Re-checked here, not just at draft time: a colleague may have resolved or already
    # queried this finding while the draft sat open in front of someone.
    _require_queryable(row)
    if row.get("query_sent_at") is not None:
        raise ValueError(f"disc:{row['discrepancy_id']} was already queried on "
                         f"{_iso(row['query_sent_at'])}")

    if not _send_email(to=to, subject=subject, body=body, agent_nick=agent_nick):
        raise RuntimeError(f"the query to {to} was not accepted for delivery")

    sent_at = datetime.now(timezone.utc)
    cur.execute(_STAMP_SQL, (sent_at, row["discrepancy_id"]))
    agent_actions.record_action(
        phase="validation", action_type="supplier_query", agent=AGENT, conn=conn,
        deal_id=row.get("deal_id"), doc_pk=row.get("doc_pk_candidate"), doc_type="invoice",
        field_name="query_sent_at", status="ok",
        summary=f"queried {row.get('supplier_name') or 'the supplier'} about "
                f"{row.get('doc_pk_candidate')}",
        details={"finding_id": f"disc:{row['discrepancy_id']}", "to": to,
                 "subject": subject, "issue_type": row.get("issue_type"),
                 "figures": figures(row)},
    )
    conn.commit()
    logger.info("value-found query sent for disc:%s to %s", row["discrepancy_id"], to)
    return {"status": "sent", "finding_id": f"disc:{row['discrepancy_id']}",
            "query_sent_at": sent_at.isoformat()}
