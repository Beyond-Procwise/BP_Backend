"""Emails to colleagues (forwards, internal replies): stored pending, sent only once the
person who wrote them approves.

Nothing here sends on creation. ``approve_and_send`` is the only way out, and it checks,
at that moment and in this order: the email exists and is still pending, the approver is
its author (by the email on their sign-in, any case), the stored content is the content
that was hashed, the email policy still allows emails to colleagues and every recipient is
still inside the company, and the company's default shared mailbox can send. Only then is
the row claimed -- a conditional ``pending -> approved`` update, so two approvals at once
cannot both send -- and the email leaves from the default shared mailbox.

A send that fails after the claim marks the row ``held`` with a plain reason. It never goes
back to ``pending`` on its own.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from services import mail_policy, mail_rules
from services.agent_actions import record_action
from services.approval_content import content_hash as _hash
from services.mail_m365.config import read_integration
from services.mail_providers import get_provider
from services.style.mailbox import HEALTH_OK, MailboxBindingRepository

logger = logging.getLogger(__name__)

KINDS = ("forward", "reply", "notice")

_NOT_AUTHOR = "Only the person who wrote this email can approve it"
_POLICY_OFF = "Emails to colleagues are switched off in your email policy"
_POLICY_UNREADABLE = "Your email policy could not be read, so nothing was sent"
_OUTSIDE = "{addr} is outside your company. Use the normal supplier email instead."
_NOT_ADDRESS = "'{raw}' is not a single email address"
_NO_RECIPIENTS = "Add at least one colleague to send this to"
_NO_ATTACHMENTS = "Emails to colleagues cannot carry attachments yet"
_ALREADY_DONE = "This email has already been approved, sent, held or declined"
_CHANGED = "This email changed after it was written, so it was not sent; open it again and approve what you see"
_NOT_FOUND = "There is no such email"

HELD_NOT_CONNECTED = "Your company mailbox is not connected, so this email is held"
HELD_NO_DEFAULT = "No default company mailbox is chosen yet, so this email is held"
HELD_MAILBOX_UNAVAILABLE = "The company mailbox {address} is not available right now, so this email is held"
HELD_TOO_LARGE = "This email is too large to send from your company mailbox (limit {mb} MB including attachments)"
HELD_REJECTED = "Your company mailbox did not accept the email, so it is held; your admin has been told"
HELD_CHANGED = "This email changed while it was being sent, so it is held"
HELD_FAILED = "Something went wrong while sending this email, so it is held and was not sent"
HELD_SENT_NOT_RECORDED = ("This email was sent (message id {message_id}) but could not be marked "
                          "as sent, so it is held; do not send it again")


class InternalEmailRefused(Exception):
    """Why an email to a colleague was not created, changed or sent. ``message`` is plain English."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_COLUMNS = ("internal_email_id, author, kind, mailbox_address, recipients, subject, body, "
            "attachments, source_ref, status, held_reason, content_hash, approved_by, "
            "approved_at, sent_at, message_id, created_at")
_NAMES = [c.strip() for c in _COLUMNS.split(",")]


def _row(r) -> Dict[str, Any]:
    out = dict(zip(_NAMES, r))
    for key in ("recipients", "attachments"):
        if isinstance(out[key], str):
            out[key] = json.loads(out[key])
    return out


def _hash_of(recipients, subject, body, attachments) -> str:
    return _hash({"recipients": list(recipients), "subject": subject, "body": body,
                  "attachments": list(attachments)})


def _same_person(a: Optional[str], b: Optional[str]) -> bool:
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    return bool(a) and a == b


def _load_policy():
    try:
        return mail_policy.load()
    except mail_policy.PolicyUnavailable:
        logger.exception("email policy unavailable for an email to a colleague")
        raise InternalEmailRefused("policy_off", _POLICY_UNREADABLE) from None


def _colleagues(policy, recipients: List[str]) -> List[str]:
    """The recipients as lower-cased bare addresses, refusing any outside the company.

    Only "internal" matters here, so no supplier lookup is made: an address that is not
    in a company domain is refused whatever else it is."""
    out: List[str] = []
    for raw in recipients or []:
        addr = mail_rules.bare_address(raw)
        if "@" not in addr or addr.startswith("@") or addr.endswith("@"):
            raise InternalEmailRefused("not_internal", _NOT_ADDRESS.format(raw=raw))
        if mail_rules.classify(addr, policy, is_supplier=lambda _a: False) != "internal":
            raise InternalEmailRefused("not_internal", _OUTSIDE.format(addr=addr))
        if addr not in out:
            out.append(addr)
    if not out:
        raise InternalEmailRefused("not_internal", _NO_RECIPIENTS)
    return out


class _SqlStore:
    """Every statement the outbox runs. Each status change names the status it moves from
    in its WHERE, so a change that lost a race changes nothing and says so (returns None)."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def _one(self, sql: str, params) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        r = cur.fetchone()
        return _row(r) if r is not None else None

    def get(self, internal_email_id: int) -> Optional[Dict[str, Any]]:
        return self._one(f"SELECT {_COLUMNS} FROM proc.bp_internal_email "
                         "WHERE internal_email_id = %s", (internal_email_id,))

    def insert(self, *, author, kind, recipients, subject, body, attachments, source_ref,
               content_hash) -> Dict[str, Any]:
        return self._one(
            f"""INSERT INTO proc.bp_internal_email
                   (author, kind, recipients, subject, body, attachments, source_ref, content_hash)
                VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, %s, %s) RETURNING {_COLUMNS}""",
            (author, kind, json.dumps(recipients), subject, body, json.dumps(attachments),
             source_ref, content_hash))

    def pending_for(self, author: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(f"""SELECT {_COLUMNS} FROM proc.bp_internal_email
                         WHERE lower(author) = lower(%s) AND status = 'pending'
                         ORDER BY created_at DESC, internal_email_id DESC""", (author,))
        return [_row(r) for r in cur.fetchall()]

    def update_content(self, internal_email_id, *, author, recipients, subject, body,
                       content_hash) -> Optional[Dict[str, Any]]:
        return self._one(
            f"""UPDATE proc.bp_internal_email
                   SET recipients = %s::jsonb, subject = %s, body = %s, content_hash = %s
                 WHERE internal_email_id = %s AND status = 'pending' AND lower(author) = lower(%s)
             RETURNING {_COLUMNS}""",
            (json.dumps(recipients), subject, body, content_hash, internal_email_id, author))

    def decline(self, internal_email_id, *, author) -> Optional[Dict[str, Any]]:
        return self._one(
            f"""UPDATE proc.bp_internal_email SET status = 'declined'
                 WHERE internal_email_id = %s AND status = 'pending' AND lower(author) = lower(%s)
             RETURNING {_COLUMNS}""", (internal_email_id, author))

    def hold(self, internal_email_id, reason, *, from_status, mailbox) -> Optional[Dict[str, Any]]:
        return self._one(
            f"""UPDATE proc.bp_internal_email
                   SET status = 'held', held_reason = %s,
                       mailbox_address = COALESCE(%s, mailbox_address)
                 WHERE internal_email_id = %s AND status = %s RETURNING {_COLUMNS}""",
            (reason, mailbox, internal_email_id, from_status))

    def claim(self, internal_email_id, *, approver, mailbox, content_hash) -> Optional[Dict[str, Any]]:
        return self._one(
            f"""UPDATE proc.bp_internal_email
                   SET status = 'approved', approved_by = %s, approved_at = now(),
                       mailbox_address = %s
                 WHERE internal_email_id = %s AND status = 'pending' AND content_hash = %s
             RETURNING {_COLUMNS}""",
            (approver, mailbox, internal_email_id, content_hash))

    def mark_sent(self, internal_email_id, *, message_id) -> Optional[Dict[str, Any]]:
        return self._one(
            f"""UPDATE proc.bp_internal_email SET status = 'sent', sent_at = now(), message_id = %s
                 WHERE internal_email_id = %s AND status = 'approved' RETURNING {_COLUMNS}""",
            (message_id, internal_email_id))


def _store(conn):
    return _SqlStore(conn)


def _get(store, internal_email_id: int) -> Dict[str, Any]:
    row = store.get(internal_email_id)
    if row is None:
        raise InternalEmailRefused("not_found", _NOT_FOUND)
    return row


def create(conn, *, author: str, kind: str, recipients: List[str], subject: str, body: str,
           attachments: List[dict], source_ref: Optional[str]) -> Dict[str, Any]:
    """Store an email to colleagues as ``pending``. Nothing is sent."""
    if not str(author or "").strip():
        raise InternalEmailRefused("not_author", "Your sign-in has no email address, so you cannot write emails to colleagues")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if attachments:
        raise ValueError(_NO_ATTACHMENTS)
    policy = _load_policy()
    if policy.internal_send != "with_user_approval":
        raise InternalEmailRefused("policy_off", _POLICY_OFF)
    addrs = _colleagues(policy, recipients)
    return _store(conn).insert(author=author.strip(), kind=kind, recipients=addrs,
                               subject=subject, body=body, attachments=[], source_ref=source_ref,
                               content_hash=_hash_of(addrs, subject, body, []))


def list_pending(conn, *, author: str) -> List[Dict[str, Any]]:
    return _store(conn).pending_for(str(author or ""))


def _check_mine_and_pending(row: Dict[str, Any], who: Optional[str], not_author_message: str) -> None:
    if row["status"] != "pending":
        raise InternalEmailRefused("already_done", _ALREADY_DONE)
    if not _same_person(who, row["author"]):
        raise InternalEmailRefused("not_author", not_author_message)


def edit(conn, internal_email_id: int, *, author: str, subject: str, body: str,
         recipients: List[str]) -> Dict[str, Any]:
    """Change a pending email. The content hash is recomputed, so an approval is always
    of the content the author last saw."""
    store = _store(conn)
    row = _get(store, internal_email_id)
    _check_mine_and_pending(row, author, "Only the person who wrote this email can change it")
    addrs = _colleagues(_load_policy(), recipients)
    updated = store.update_content(
        internal_email_id, author=author, recipients=addrs, subject=subject, body=body,
        content_hash=_hash_of(addrs, subject, body, row["attachments"]))
    if updated is None:  # approved, declined or held in between
        raise InternalEmailRefused("already_done", _ALREADY_DONE)
    return updated


def decline(conn, internal_email_id: int, *, actor: str) -> Dict[str, Any]:
    store = _store(conn)
    row = _get(store, internal_email_id)
    _check_mine_and_pending(row, actor, "Only the person who wrote this email can decline it")
    declined = store.decline(internal_email_id, author=actor)
    if declined is None:
        raise InternalEmailRefused("already_done", _ALREADY_DONE)
    return declined


def _sending_mailbox(conn, provider):
    """(provider, default mailbox, None) or (None, mailbox-or-None, plain held reason)."""
    provider = provider if provider is not None else get_provider(conn)
    if provider is None or read_integration(conn).get("status") != "connected":
        return None, None, HELD_NOT_CONNECTED
    binding = MailboxBindingRepository(conn).get_default_sender()
    if binding is None:
        return None, None, HELD_NO_DEFAULT
    if not (binding.is_active and binding.health_state == HEALTH_OK and binding.can_send):
        return None, binding, HELD_MAILBOX_UNAVAILABLE.format(address=binding.mailbox_address)
    return provider, binding, None


def _audit(action_type: str, row: Dict[str, Any], *, status: str, summary: str, **extra) -> None:
    record_action(phase="send", action_type=action_type, agent="internal_email", status=status,
                  summary=summary,
                  details={"internal_email_id": row["internal_email_id"], "author": row["author"],
                           "recipients": row["recipients"], **extra})


def approve_and_send(conn, internal_email_id: int, *, approver: str, provider=None) -> Dict[str, Any]:
    """Approve and send, if and only if every check passes now. See the module docstring."""
    store = _store(conn)
    row = _get(store, internal_email_id)
    if row["status"] != "pending":
        raise InternalEmailRefused("already_done", _ALREADY_DONE)
    if not _same_person(approver, row["author"]):
        raise InternalEmailRefused("not_author", _NOT_AUTHOR)
    pinned = row["content_hash"]
    if _hash_of(row["recipients"], row["subject"], row["body"], row["attachments"]) != pinned:
        raise InternalEmailRefused("changed", _CHANGED)

    # The policy as it is now, not as it was when the email was written.
    policy = _load_policy()
    if policy.internal_send != "with_user_approval":
        raise InternalEmailRefused("policy_off", _POLICY_OFF)
    _colleagues(policy, row["recipients"])

    provider, binding, reason = _sending_mailbox(conn, provider)
    if reason is not None:
        mailbox = binding.mailbox_address if binding is not None else None
        if store.hold(internal_email_id, reason, from_status="pending", mailbox=mailbox) is None:
            raise InternalEmailRefused("already_done", _ALREADY_DONE)
        _audit("internal_email_held", row, status="held", summary=reason)
        raise InternalEmailRefused("no_mailbox", reason)

    # Claim the row. Only one approval can move it out of pending, and only with the
    # content whose hash was checked above.
    claimed = store.claim(internal_email_id, approver=approver.strip(),
                          mailbox=binding.mailbox_address, content_hash=pinned)
    if claimed is None:
        raise InternalEmailRefused("already_done", _ALREADY_DONE)
    row = claimed

    def held(reason: str, **extra) -> Dict[str, Any]:
        out = store.hold(internal_email_id, reason, from_status="approved", mailbox=None)
        _audit("internal_email_held", row, status="held", summary=reason, **extra)
        return out if out is not None else _get(store, internal_email_id)

    if _hash_of(row["recipients"], row["subject"], row["body"], row["attachments"]) != pinned:
        return held(HELD_CHANGED)

    provider_accepted = False
    try:
        from services.email_service import build_mime

        msg, message_id = build_mime(row["subject"], row["body"], row["recipients"],
                                     binding.mailbox_address)
        mime = msg.as_bytes()
        limit = provider.size_limit()
        if len(mime) > limit:
            return held(HELD_TOO_LARGE.format(mb=limit // 1_000_000), size=len(mime))
        try:
            provider.send(binding, mime)
        except Exception as exc:  # noqa: BLE001 - the raw cause is for the log; the author gets a sentence
            logger.error("Email to colleagues %s from %s failed: %s",
                         internal_email_id, binding.mailbox_address, exc)
            return held(HELD_REJECTED)
        provider_accepted = True
        sent = store.mark_sent(internal_email_id, message_id=message_id)
        if sent is None:  # cannot happen while this call holds the claim; never report a false state
            sent = _get(store, internal_email_id)
    except Exception as exc:  # noqa: BLE001 - any failure after the claim holds the row, never leaves it approved
        logger.error("Email to colleagues %s failed after it was claimed: %s",
                     internal_email_id, exc)
        if provider_accepted:  # the mail has gone; never send it again
            return held(HELD_SENT_NOT_RECORDED.format(message_id=message_id))
        return held(HELD_FAILED)
    _audit("internal_email_sent", sent, status="sent",
           summary=f"email to colleagues sent from {binding.mailbox_address}",
           approved_by=sent["approved_by"], message_id=message_id)
    return sent
