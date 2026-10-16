"""Mail arriving in a company shared mailbox.

Two readers, each with its own cursor, both read-only: mail-intake takes every message
(attachments -> quarantine/dedup -> extraction) and the email watcher takes them as
possible supplier replies. Neither moves, flags or marks anything in the mailbox.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

from services.mail_intake.sources.base import RawMessage
from services.mail_m365.graph_client import M365Client, Throttled

log = logging.getLogger(__name__)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


_MSGID = re.compile(r"<[^<>\s]+>")
NARROW_SCOPE = "replies_and_suppliers"


def _read_scope() -> str:
    """The policy's read scope; an unreadable policy reads the narrower scope."""
    try:
        from services.mail_policy import load
        return load().read_scope
    except Exception:
        log.warning("email policy unreadable; reading only replies and supplier mail", exc_info=True)
        return NARROW_SCOPE


def _default_checks() -> tuple:
    """(is_supplier, sent_ids) backed by the database. A failed lookup RAISES: the caller
    must be able to tell "could not check" from "no"."""
    from services.db import get_conn

    def is_supplier(address: str) -> bool:
        from services.supplier_contact import is_known_supplier_email
        with get_conn() as c:
            return bool(address) and is_known_supplier_email(c, address)

    def sent_ids(ids: List[str]) -> bool:
        with get_conn() as c:
            cur = c.cursor()
            cur.execute("SELECT 1 FROM proc.email_thread_map WHERE message_id = ANY(%s) LIMIT 1",
                        (ids,))
            return cur.fetchone() is not None

    return is_supplier, sent_ids


def scope_state(item: Mapping[str, Any], *, policy: Any, client: Any, mailbox: str,
                is_supplier: Callable[[str], bool], sent_ids: Callable[[List[str]], bool]
                ) -> Optional[bool]:
    """True = download; False = definitely out of scope; None = could not check (retry
    next pass). Either way a non-True message is never passed to get_mime."""
    if policy.read_scope == "whole_inbox":
        return True
    unknown = False
    try:
        if is_supplier(item.get("from") or ""):
            return True
    except Throttled:
        raise
    except Exception:
        log.warning("supplier lookup failed for %s; will retry", item.get("id"), exc_info=True)
        unknown = True
    try:
        headers = client.get_headers(mailbox, item["id"])
        found = _MSGID.findall(headers.get("in-reply-to", "") + " " + headers.get("references", ""))
        ids: List[str] = []
        for m in found:
            ids += [m, m.strip("<>")]
        if ids and sent_ids(ids):
            return True
    except Throttled:
        raise
    except Exception:
        log.warning("m365 scope check failed for %s; message left unread, will retry",
                    item.get("id"), exc_info=True)
        return None
    return None if unknown else False


def in_scope(item: Mapping[str, Any], *, policy: Any, client: Any, mailbox: str,
             is_supplier: Callable[[str], bool], sent_ids: Callable[[List[str]], bool]) -> bool:
    """May this Inbox message be downloaded? Decided from the sender and two headers only;
    an out-of-scope or unchecked message is never passed to get_mime."""
    return scope_state(item, policy=policy, client=client, mailbox=mailbox,
                       is_supplier=is_supplier, sent_ids=sent_ids) is True


class GraphMailSource:
    def __init__(self, client_for: Callable[[], M365Client]) -> None:
        self._client_for = client_for

    def iter_messages(self, binding: Mapping[str, Any], since: datetime) -> Iterator[RawMessage]:
        """In-scope Inbox messages, oldest first (list_inbox orders by arrival).

        A message whose scope check could not be completed ends the listing there: the
        caller moves the binding's cursor to the newest message it commits, so yielding
        anything after an unchecked message would carry the cursor past it and it would
        never be read. The unchecked one is not downloaded and is checked again next pass.
        A definite "out of scope" is skipped as before.
        """
        mailbox = binding["mailbox_address"]
        client = self._client_for()
        policy = SimpleNamespace(read_scope=_read_scope())
        is_supplier, sent_ids = _default_checks()
        for item in client.list_inbox(mailbox, since):
            state = scope_state(item, policy=policy, client=client, mailbox=mailbox,
                                is_supplier=is_supplier, sent_ids=sent_ids)
            if state is None:
                log.warning("m365 intake for %s stops before message %s: its scope could not "
                            "be checked; the cursor stays before it and it is retried next pass",
                            mailbox, item.get("id"))
                return
            if state is False:
                # Graph id and binding only: no sender, subject or content in the log.
                log.info("m365 intake skipped out-of-scope message %s (binding %s)",
                         item.get("id"), binding.get("binding_id"))
                continue
            yield RawMessage(raw=client.get_mime(mailbox, item["id"]),
                             provider_message_id=item["id"],
                             received_at=_parse_dt(item.get("receivedDateTime")),
                             source_ref=f"m365:{mailbox}")


CLIENT_TTL_SECONDS = 600
SEEN_LIMIT = 4000

_lock = threading.Lock()
_client_cache: Dict[str, Any] = {}
# (mailbox, graph id) -> the parsed reply. The watcher asks every ~30 s, once per active
# workflow; a message already downloaded is not downloaded again, and is handed to every
# workflow's pass (not just the first) so none of them misses a reply that belongs to it.
_parsed: "OrderedDict[tuple, Any]" = OrderedDict()
# Messages judged out of scope: not looked at again every pass.
_skipped: "OrderedDict[tuple, bool]" = OrderedDict()


def _reset_caches() -> None:
    with _lock:
        _client_cache.clear()
        _parsed.clear()
        _skipped.clear()


def _default_client_for() -> Optional[Callable[[], M365Client]]:
    """A client factory for the connected tenant, or None when Microsoft 365 is not set up.

    The client (Secrets Manager read + token) is built once per process and reused for
    CLIENT_TTL_SECONDS; a disconnect or reconnect is picked up within that time.
    """
    from services.mail_m365.config import load_settings
    settings = load_settings()
    if settings is None:
        return None
    now = time.monotonic()
    with _lock:
        hit = _client_cache.get("c")
        if hit and now - hit["at"] < CLIENT_TTL_SECONDS and hit["settings"] == settings:
            client = hit["client"]
            return lambda: client
    from services.db import get_conn
    from services.mail_m365.admin import _default_factory
    from services.mail_m365.config import read_integration
    with get_conn() as c:
        integration = read_integration(c)
    if integration.get("status") != "connected" or not integration.get("tenant_id"):
        return None
    client = _default_factory(settings)(integration["tenant_id"])
    with _lock:
        _client_cache["c"] = {"at": now, "client": client, "settings": settings,
                              "tenant": integration["tenant_id"]}
    return lambda: client


def _replies_from(mailbox: str, client: Any, since: datetime, parse: Callable[[bytes], Any]) -> List[Any]:
    out: List[Any] = []
    policy = SimpleNamespace(read_scope=_read_scope())
    is_supplier, sent_ids = _default_checks()
    for item in client.list_inbox(mailbox, since):
        key = (mailbox, item["id"])
        with _lock:
            cached = _parsed.get(key)
            if cached is not None:
                _parsed.move_to_end(key)
            skipped = key in _skipped
        if cached is None and skipped and policy.read_scope != "whole_inbox":
            continue
        if cached is None:
            state = scope_state(item, policy=policy, client=client, mailbox=mailbox,
                                is_supplier=is_supplier, sent_ids=sent_ids)
            if state is not True:
                if state is False:      # only a definite no is remembered; "unknown" is retried
                    with _lock:
                        _skipped[key] = True
                        while len(_skipped) > SEEN_LIMIT:
                            _skipped.popitem(last=False)
                continue
            cached = parse(client.get_mime(mailbox, item["id"]))
            with _lock:
                _parsed[key] = cached
                while len(_parsed) > SEEN_LIMIT:
                    _parsed.popitem(last=False)
        out.append(dataclasses.replace(cached))
    return out


def fetch_replies(since: datetime, *, repo: Any = None,
                  client_for: Optional[Callable[[], Any]] = None) -> List[Any]:
    """Possible supplier replies from every active shared mailbox. Never raises."""
    out: List[Any] = []
    try:
        from services.email_watcher import parse_email_bytes
        from services.style.mailbox import PROVIDER_GRAPH, MailboxBindingRepository
        repo = repo or MailboxBindingRepository()
        bindings = [b for b in repo.list_org()
                    if b.provider == PROVIDER_GRAPH and b.can_send and b.usable]
        if not bindings:
            return out
        client_for = client_for or _default_client_for()
        if client_for is None:
            return out
        client = client_for()
        for b in bindings:
            try:
                out.extend(_replies_from(b.mailbox_address, client, since, parse_email_bytes))
            except Exception:
                log.warning("m365 reply fetch failed for %s; other mailboxes continue",
                            b.mailbox_address, exc_info=True)
    except Exception:
        log.warning("m365 reply fetch failed; returning what was read", exc_info=True)
    return out


def has_bindings() -> bool:
    """True when a Graph shared mailbox is connected (for the watcher's 'nothing configured' warning)."""
    try:
        from services.style.mailbox import PROVIDER_GRAPH, MailboxBindingRepository
        return any(b.provider == PROVIDER_GRAPH and b.can_send and b.usable
                   for b in MailboxBindingRepository().list_org())
    except Exception:
        return False


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


def _mark_ok_if_degraded(repo: Any, conn: Any, binding_id: Any) -> None:
    """The Inbox was just listed successfully, which is exactly what the hourly health
    probe checks: a mailbox left DEGRADED by one 429/5xx is healthy again now, not in up to
    an hour (sending holds while DEGRADED). Only DEGRADED and active is lifted: REVOKED or
    removed is the customer's or an admin's decision, which a poll never reverses."""
    try:
        from services.style.mailbox import HEALTH_DEGRADED, HEALTH_OK
        b = repo.get(binding_id)
        if b is not None and b.is_active and b.health_state == HEALTH_DEGRADED:
            repo.record_health(binding_id, HEALTH_OK)
            if conn is not None and not getattr(conn, "autocommit", False):
                conn.commit()
            log.info("m365 mailbox %s answered the poll; DEGRADED -> OK", b.mailbox_address)
    except Exception:  # health bookkeeping never breaks the poll
        log.debug("m365 health recovery skipped for binding=%s", binding_id, exc_info=True)


def poll_m365(conn: Any, pipeline: Any, *, repo: Any = None) -> Dict[str, int]:
    from services.mail_intake.poll import active_bindings
    with conn.cursor() as cur:
        bindings = active_bindings(cur, provider="graph", roles=("send_receive",))
    conn.commit()
    if not bindings:
        return {"bindings": 0}
    client_for = _default_client_for()
    if client_for is None:
        return {"bindings": 0}
    if repo is None:
        from services.style.mailbox import MailboxBindingRepository
        repo = MailboxBindingRepository(conn)
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
        if not counts.get("throttled"):
            _mark_ok_if_degraded(repo, conn, binding.get("binding_id"))
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    return total
