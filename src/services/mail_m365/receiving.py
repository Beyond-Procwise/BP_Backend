"""Mail arriving in a company shared mailbox.

Two readers, each with its own cursor, both read-only: mail-intake takes every message
(attachments -> quarantine/dedup -> extraction) and the email watcher takes them as
possible supplier replies. Neither moves, flags or marks anything in the mailbox.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from services.mail_intake.sources.base import RawMessage
from services.mail_m365.graph_client import M365Client, Throttled

log = logging.getLogger(__name__)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


class GraphMailSource:
    def __init__(self, client_for: Callable[[], M365Client]) -> None:
        self._client_for = client_for

    def iter_messages(self, binding: Mapping[str, Any], since: datetime) -> Iterator[RawMessage]:
        mailbox = binding["mailbox_address"]
        client = self._client_for()
        for item in client.list_inbox(mailbox, since):
            yield RawMessage(raw=client.get_mime(mailbox, item["id"]),
                             provider_message_id=item["id"],
                             received_at=_parse_dt(item.get("receivedDateTime")),
                             source_ref=f"m365:{mailbox}")


def _default_client_for() -> Optional[Callable[[], M365Client]]:
    """A client factory for the connected tenant, or None when Microsoft 365 is not set up."""
    from services.db import get_conn
    from services.mail_m365.admin import _default_factory
    from services.mail_m365.config import load_settings, read_integration
    settings = load_settings()
    if settings is None:
        return None
    with get_conn() as c:
        integration = read_integration(c)
    if integration.get("status") != "connected" or not integration.get("tenant_id"):
        return None
    client = _default_factory(settings)(integration["tenant_id"])
    return lambda: client


def fetch_replies(since: datetime, *, repo: Any = None,
                  client_for: Optional[Callable[[], Any]] = None) -> List[Any]:
    """Possible supplier replies from every active shared mailbox. Never raises."""
    out: List[Any] = []
    try:
        from services.email_watcher import parse_email_bytes
        from services.style.mailbox import MailboxBindingRepository
        repo = repo or MailboxBindingRepository()
        bindings = [b for b in repo.list_org() if b.can_send and b.usable]
        if not bindings:
            return out
        client_for = client_for or _default_client_for()
        if client_for is None:
            return out
        for b in bindings:
            try:
                for raw in GraphMailSource(client_for).iter_messages(
                        {"mailbox_address": b.mailbox_address}, since):
                    out.append(parse_email_bytes(raw.raw))
            except Exception:
                log.warning("m365 reply fetch failed for %s; other mailboxes continue",
                            b.mailbox_address, exc_info=True)
    except Exception:
        log.warning("m365 reply fetch failed; returning what was read", exc_info=True)
    return out


def poll_one(conn: Any, binding: Mapping[str, Any], source: GraphMailSource,
             pipeline: Any) -> Dict[str, int]:
    from services.mail_intake.poll import poll_binding
    try:
        return poll_binding(conn, binding, source, pipeline)
    except Throttled as exc:
        # Raised while listing, before any message is committed, so the binding's cursor
        # (last_sync_at) is where it was and the next pass fetches the same messages.
        log.info("m365 poll throttled for %s; retry after ~%ss, cursor unchanged",
                 binding.get("mailbox_address"), exc.retry_after)
        if conn is not None:
            conn.rollback()
        return {"throttled": 1}


def poll_m365(conn: Any, pipeline: Any) -> Dict[str, int]:
    from services.mail_intake.poll import active_bindings
    with conn.cursor() as cur:
        bindings = active_bindings(cur, provider="graph", roles=("send_receive",))
    conn.commit()
    if not bindings:
        return {"bindings": 0}
    client_for = _default_client_for()
    if client_for is None:
        return {"bindings": 0}
    source = GraphMailSource(client_for)
    total: Dict[str, int] = {"bindings": len(bindings)}
    for binding in bindings:
        try:
            counts = poll_one(conn, binding, source, pipeline)
        except Exception:
            conn.rollback()
            log.exception("m365 poll binding=%s failed", binding.get("binding_id"))
            total["binding_errors"] = total.get("binding_errors", 0) + 1
            continue
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    return total
