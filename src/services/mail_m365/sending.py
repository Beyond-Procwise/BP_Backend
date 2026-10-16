"""Which way a supplier email leaves.

Not connected (and never was) -> SES, exactly as before. Connected and the default mailbox
healthy -> Microsoft 365, from that mailbox. Anything in between -> HELD. A supplier who has
been writing to procurement@customer.com and suddenly hears from a Procwise address is right
to be suspicious, so a broken company mailbox stops sending rather than changing sender
(spec 5.1).

Held reasons are read by buyers (compose, the draft), so they are plain sentences: no
health-state codes, no Microsoft error text. The raw detail goes to the log.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from services.mail_m365.graph_client import MessageTooLarge
from services.style.graph_source import MailboxUnreachable
from services.style.mailbox import (HEALTH_DEGRADED, HEALTH_OK, HEALTH_REVOKED,
                                    MailboxBinding, MailboxBindingRepository)

logger = logging.getLogger(__name__)

# Graph's sendMail takes the MIME base64-encoded in a request capped at about 4 MB, so the
# raw message (attachments included) must stay under 3 MB. Larger needs an upload session,
# which is a follow-up; until then such an email is held with this reason.
MAX_MIME_BYTES = 3_000_000

HELD_TOO_LARGE = ("This email is too large to send from your Microsoft 365 mailbox "
                  "(limit 3 MB including attachments)")
HELD_REJECTED = "Microsoft 365 did not accept the email; your admin has been told"
HELD_UNCHECKED = "The company mailbox could not be checked, so this email is held; try again shortly"
HELD_DEGRADED = "The company mailbox {address} is temporarily unavailable, so this email is held"
HELD_REVOKED = "Access to {address} was withdrawn in Microsoft 365, so this email is held"
HELD_INACTIVE = ("The company mailbox {address} is no longer in use, so this email is held "
                 "until an admin chooses a default mailbox")
HELD_DISCONNECTED = ("Microsoft 365 is disconnected, so this email is held rather than sent "
                     "from another address")
HELD_NO_DEFAULT = ("Microsoft 365 is connected but no default mailbox is chosen yet, so this "
                   "email is held")


@dataclass(frozen=True)
class Decision:
    kind: str                       # "ses" | "graph" | "held"
    binding: Optional[MailboxBinding]
    reason: Optional[str]
    # True when the email goes out from the Procwise address although Microsoft 365 was
    # connected at some point: the supplier-email recipient rules still apply to it.
    fallback: bool = False


def _never_connected(integration: Dict[str, Any]) -> bool:
    """No provider was ever recorded: Microsoft 365 was never connected here (a Connect
    that was started but not finished leaves only pending nonces)."""
    return not integration.get("provider")


def decide(conn: Any = None, *, repo: Any = None,
           integration: Optional[Dict[str, Any]] = None) -> Decision:
    # A customer that never connected Microsoft 365 goes to SES without the binding
    # columns being read at all, so a database the m365 migration has not reached yet
    # cannot hold every customer's supplier email.
    if integration is not None and _never_connected(integration):
        return Decision("ses", None, None)
    if repo is not None and integration is not None:
        return _decide(repo, integration)
    from services.mail_m365.config import read_integration
    if conn is not None:
        return _decide_on(conn, repo, integration, read_integration)
    from services.db import get_conn
    with get_conn() as c:
        return _decide_on(c, repo, integration, read_integration)


def _decide_on(conn: Any, repo: Any, integration: Optional[Dict[str, Any]],
               read_integration: Callable[[Any], Dict[str, Any]]) -> Decision:
    integration = read_integration(conn) if integration is None else integration
    if _never_connected(integration):
        return Decision("ses", None, None)
    return _decide(repo or MailboxBindingRepository(conn), integration)


def _reason_for(binding: MailboxBinding) -> str:
    if binding.health_state == HEALTH_REVOKED:
        return HELD_REVOKED.format(address=binding.mailbox_address)
    if binding.health_state == HEALTH_DEGRADED:
        return HELD_DEGRADED.format(address=binding.mailbox_address)
    return HELD_INACTIVE.format(address=binding.mailbox_address)


def _policy_sends_from_procwise() -> bool:
    """The email policy's ``on_disconnect`` choice. Only asked when the company mailbox
    cannot be used; an unreadable policy means held, as it always was."""
    from services import mail_policy
    try:
        return mail_policy.load().on_disconnect == "procwise_address"
    except Exception as exc:  # noqa: BLE001 - cannot tell what the company chose: hold
        logger.warning("Email policy unreadable while choosing the mail route: %s", exc)
        return False


def _decide(repo: Any, integration: Dict[str, Any]) -> Decision:
    if _never_connected(integration):
        return Decision("ses", None, None)
    default = repo.get_default_sender()
    connected = integration.get("status") == "connected"
    if default is not None and connected:
        if default.is_active and default.health_state == HEALTH_OK:
            return Decision("graph", default, None)
        if default.health_state != HEALTH_DEGRADED and _policy_sends_from_procwise():
            return Decision("ses", None, None, True)   # access withdrawn / mailbox deactivated
        return Decision("held", default, _reason_for(default))
    # No usable default. Has the company ever sent from its own mailbox?
    ever = default or repo.get_org_default_any()
    if ever is not None:
        if (not connected or ever.health_state != HEALTH_DEGRADED) and _policy_sends_from_procwise():
            return Decision("ses", None, None, True)
        return Decision("held", ever, _reason_for(ever) if connected else HELD_DISCONNECTED)
    if connected:
        if _policy_sends_from_procwise():
            return Decision("ses", None, None, True)
        return Decision("held", None, HELD_NO_DEFAULT)
    # Connected once, disconnected before any test send: nothing was ever sent from the
    # company's mailbox, so the Procwise sender is what suppliers have seen.
    return Decision("ses", None, None, True)


def held_reason_for(exc: BaseException) -> str:
    """The sentence a buyer sees when a Microsoft 365 send fails. The caller logs exc."""
    return HELD_TOO_LARGE if isinstance(exc, MessageTooLarge) else HELD_REJECTED


def send_via_graph(binding: MailboxBinding, mime: bytes, *,
                   client_for: Optional[Callable[[], Any]] = None) -> None:
    """Send through the binding's mailbox. Raises on any failure; the caller holds.

    The size is checked before anything else (no secret read, no request): Graph would
    refuse it anyway, with an error a buyer could not act on. The client is the cached
    one the readers use, so a send is not a Secrets Manager read and a token request."""
    if len(mime) > MAX_MIME_BYTES:
        raise MessageTooLarge(len(mime), MAX_MIME_BYTES)
    if client_for is None:
        from services.mail_m365.receiving import _default_client_for
        client_for = _default_client_for()
    if client_for is None:
        raise MailboxUnreachable("Microsoft 365 is not configured or not connected")
    client_for().send_mime(binding.mailbox_address, mime)
