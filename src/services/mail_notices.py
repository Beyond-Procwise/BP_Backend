"""Product notices (the weekly value digest and similar): which address do they leave from?

Today every notice leaves from the Procwise address. When the company's email policy says
``notices_from_mailbox = with_user_approval``, the company mailbox is connected with a
default mailbox, and every recipient is a colleague, the notice is stored as a pending
internal email for the user it is for, and goes out from the shared mailbox only once that
user approves it. In every other case -- policy off, not connected, policy unreadable,
anyone outside the company, or anything going wrong while storing it -- the notice goes
from the Procwise address exactly as before. A notice is never lost for want of a policy.

Alerts about the email connection itself do not use this: they always come from Procwise.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from services import internal_email, mail_policy, mail_rules
from services.mail_m365.config import read_integration
from services.style.mailbox import MailboxBindingRepository

logger = logging.getLogger(__name__)

PROCWISE = {"route": "procwise_address"}


def _mailbox_ready(conn) -> bool:
    if read_integration(conn).get("status") != "connected":
        return False
    return MailboxBindingRepository(conn).get_default_sender() is not None


def _all_internal(policy, recipients: List[str]) -> bool:
    if not recipients:
        return False
    for raw in recipients:
        addr = mail_rules.bare_address(raw)
        if "@" not in addr or mail_rules.classify(addr, policy, is_supplier=lambda _a: False) != "internal":
            return False
    return True


def _looks_like_email(value) -> bool:
    local, at, domain = str(value or "").strip().partition("@")
    return bool(at and local and domain and "@" not in domain)


def _via_mailbox(conn, for_user, recipients, subject, body, source_ref):
    # Only a real sign-in address can ever see and approve a queued notice.
    if not _looks_like_email(for_user):
        return None
    try:
        policy = mail_policy.load()
        if policy.notices_from_mailbox != "with_user_approval":
            return None
        if not _all_internal(policy, recipients) or not _mailbox_ready(conn):
            return None
        row = internal_email.create(conn, author=for_user, kind="notice", recipients=recipients,
                                    subject=subject, body=body, attachments=[],
                                    source_ref=source_ref)
        return {"route": "pending_approval", "internal_email_id": row["internal_email_id"]}
    except Exception:  # unreadable policy, refused, database trouble: fall back, never lose it
        logger.warning("notice could not be queued for approval; sending from the Procwise address",
                       exc_info=True)
        return None


def notify(conn, *, for_user: str, recipients: List[str], subject: str, body: str,
           source_ref: str, email_service, sender: str = None, conn_factory=None) -> Dict[str, Any]:
    """``conn`` may be given, or ``conn_factory`` (e.g. ``get_conn``) to open one only here.
    A connection that cannot be opened means the Procwise-address send, never no send."""
    queued = None
    try:
        if conn is not None:
            queued = _via_mailbox(conn, for_user, recipients, subject, body, source_ref)
        elif conn_factory is not None:
            with conn_factory() as own:
                queued = _via_mailbox(own, for_user, recipients, subject, body, source_ref)
    except Exception:
        logger.warning("could not reach the database for a notice; sending from the Procwise address",
                       exc_info=True)
        queued = None
    if queued is not None:
        return queued
    kwargs = {"subject": subject, "body": body, "recipients": recipients}
    if sender:
        kwargs["sender"] = sender
    result = email_service.send_email(**kwargs)
    out = dict(PROCWISE)
    out["accepted"] = bool(getattr(result, "success", result))
    return out
