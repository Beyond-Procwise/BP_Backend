"""Glue between the product and the people package: safe emit, rule source, service assembly, worker.

emit_safe is what approval code calls. It must never be able to break the action it announces:
the write runs inside a SAVEPOINT (so an error cannot abort the caller's transaction) and every
failure is logged and swallowed. DOMAIN_EVENTS=off turns emission off.
"""
from __future__ import annotations

import logging
import os
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from services.people import realtime
from services.people.events import DomainEvent, emit_on_cursor
from services.people.hierarchy import HierarchyService
from services.people.models import EntityRef, Person
from services.people.notifications import (NotificationService, PgNotificationRepo, PgOutbox, drain)
from services.people.resolver import RecipientResolver
from services.people.stores import PgPeopleStore

logger = logging.getLogger(__name__)


def events_enabled() -> bool:
    return os.environ.get("DOMAIN_EVENTS", "on").strip().lower() != "off"


def emit_safe(cur, event: DomainEvent) -> bool:
    """Emit on the caller's cursor inside its transaction; never raises."""
    if not events_enabled():
        return False
    try:
        cur.execute("SAVEPOINT domain_event")
    except Exception:
        # autocommit connection: no enclosing transaction to protect, a failed single INSERT is harmless
        try:
            return emit_on_cursor(cur, event)
        except Exception:
            logger.warning("domain event %s not recorded", event.type, exc_info=True)
            return False
    try:
        ok = emit_on_cursor(cur, event)
        cur.execute("RELEASE SAVEPOINT domain_event")
        return ok
    except Exception:
        logger.warning("domain event %s not recorded; the action is unaffected", event.type, exc_info=True)
        try:
            cur.execute("ROLLBACK TO SAVEPOINT domain_event")
        except Exception:
            logger.exception("could not roll back to the domain-event savepoint")
        return False


class PgRuleSource:
    """bp_routing_rule: the tenant's row wins over the 'default' tenant's."""

    def __init__(self, conn):
        self._conn = conn

    def targets_for(self, tenant_id: str, event_type: str) -> Optional[Tuple[List[str], bool]]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT targets, notify_actor FROM proc.bp_routing_rule "
                        "WHERE event_type=%s AND tenant_id IN (%s,'default') "
                        "ORDER BY (tenant_id=%s) DESC LIMIT 1", (event_type, tenant_id, tenant_id))
            r = cur.fetchone()
        return (list(r[0]), bool(r[1])) if r else None


def entity_for_event(event: DomainEvent) -> EntityRef:
    d = event.data or {}
    try:
        amount = Decimal(str(d["amount"])) if d.get("amount") is not None else None
    except (InvalidOperation, ValueError):
        amount = None
    return EntityRef(event.entity_type, event.entity_id, owner=d.get("owner"), requester=d.get("requester"),
                     category_owner=d.get("category_owner"), team=tuple(d.get("team") or ()),
                     watchers=tuple(d.get("watchers") or ()), amount=amount)


def _can_view(conn):
    def can_view(person: Person, entity: EntityRef) -> bool:
        if entity.entity_type == "conversation":
            # a message notification reaches current participants only, however it was routed
            from services.people.messaging import is_participant
            with conn.cursor() as cur:
                return person.is_active and is_participant(cur, int(entity.entity_id), person.subject)
        return _allow_same_tenant(person, entity)
    return can_view


def _allow_same_tenant(person: Person, entity: EntityRef) -> bool:
    # TODO(phase 5): defer to the real per-entity visibility rules (deal_visibility). Until then
    # notification text is value-free (no amounts, no supplier names) so a wrong recipient learns
    # only that an approval exists.
    return person.is_active


def _prefs_from_db(conn):
    def prefs_for(tenant_id: str, subject: str) -> Dict[str, Any]:
        with conn.cursor() as cur:
            cur.execute("SELECT pref_value FROM proc.bp_user_preference "
                        "WHERE user_subject=%s AND pref_key='notifications'", (subject,))
            r = cur.fetchone()
        return dict(r[0]) if r and isinstance(r[0], dict) else {}
    return prefs_for


def _graph_suggester(conn):
    """Tiers 2-4, advisory, and only handed to the resolver when the tenant switched them on."""
    from services.people.graph import queries, service
    def suggest(actor: Person, event_type: str, entity: EntityRef):
        d = {}
        key = f"{entity.entity_type}:{entity.entity_id}"
        rows = queries.suggest_recipients(service.get_store(), actor.tenant_id, actor.subject, entity_key=key,
                                          graph_tiers=True, k=5)
        return [r["subject"] for r in rows if r["tier"] >= 2]
    return suggest


def build_service(conn) -> NotificationService:
    store = PgPeopleStore(conn)
    suggester = None
    try:
        from services.people.graph import service as graph_service
        if graph_service.tiers_enabled(conn):
            suggester = _graph_suggester(conn)
    except Exception:
        suggester = None
    resolver = RecipientResolver(store, HierarchyService(store), PgRuleSource(conn), _can_view(conn), suggester)
    return NotificationService(
        resolver, PgNotificationRepo(conn), entity_for=entity_for_event,
        person_for=store.get, prefs_for=_prefs_from_db(conn),
        link_for=lambda e: (e.data or {}).get("link") or f"/{e.entity_type}/{e.entity_id}",
        on_created=lambda subject, e: realtime.push([subject], {"type": "notification", "event": e.type}))


def drain_once(limit: int = 50) -> Dict[str, int]:
    """One worker pass over the outbox (scheduled every few seconds). Email/Teams channels are not
    wired yet: only in-app notifications are produced."""
    from services.db import get_conn
    with get_conn() as conn:
        return drain(PgOutbox(conn), build_service(conn), limit)
