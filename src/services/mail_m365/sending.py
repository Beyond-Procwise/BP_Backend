"""Which way a supplier email leaves.

Not connected (and never was) -> SES, exactly as before. Connected and the default mailbox
healthy -> Microsoft 365, from that mailbox. Anything in between -> HELD. A supplier who has
been writing to procurement@customer.com and suddenly hears from a Procwise address is right
to be suspicious, so a broken company mailbox stops sending rather than changing sender
(spec 5.1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from services.style.mailbox import HEALTH_OK, MailboxBinding, MailboxBindingRepository

logger = logging.getLogger(__name__)

_PREFIX = "company mailbox unavailable: "


@dataclass(frozen=True)
class Decision:
    kind: str                       # "ses" | "graph" | "held"
    binding: Optional[MailboxBinding]
    reason: Optional[str]


def decide(conn: Any = None, *, repo: Any = None,
           integration: Optional[Dict[str, Any]] = None) -> Decision:
    if repo is not None and integration is not None:
        return _decide(repo, integration)
    from services.db import get_conn
    from services.mail_m365.config import read_integration
    with get_conn() as c:
        integration = read_integration(c) if integration is None else integration
        repo = repo or MailboxBindingRepository(c)
        return _decide(repo, integration)


def _decide(repo: Any, integration: Dict[str, Any]) -> Decision:
    default = repo.get_default_sender()
    connected = integration.get("status") == "connected"
    if default is not None and connected:
        if default.is_active and default.health_state == HEALTH_OK:
            return Decision("graph", default, None)
        return Decision("held", default,
                        f"{_PREFIX}{default.mailbox_address} is {default.health_state}")
    # No usable default. Has the company ever sent from its own mailbox?
    ever = default or repo.get_org_default_any()
    if ever is not None:
        if connected:
            state = ever.health_state if ever.health_state != HEALTH_OK else "inactive"
        else:
            state = "disconnected"
        return Decision("held", ever, f"{_PREFIX}{ever.mailbox_address} is {state}")
    if connected:
        return Decision("held", None, f"{_PREFIX}Microsoft 365 is connected but no default "
                                      "mailbox is set")
    return Decision("ses", None, None)


def send_via_graph(binding: MailboxBinding, mime: bytes, *,
                   client_factory: Optional[Callable[[str], Any]] = None,
                   settings: Any = None) -> None:
    """Send through the binding's mailbox. Raises on any failure; the caller holds."""
    from services.db import get_conn
    from services.mail_m365.admin import _default_factory
    from services.mail_m365.config import load_settings, read_integration
    settings = settings or load_settings()
    if settings is None:
        raise RuntimeError("Microsoft 365 is not configured on this server")
    with get_conn() as c:
        tenant = read_integration(c).get("tenant_id")
    client = (client_factory or _default_factory(settings))(tenant)
    client.send_mime(binding.mailbox_address, mime)
