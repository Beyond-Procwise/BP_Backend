"""NotificationService: event -> RecipientResolver -> notification rows -> channels.

In-app is always written. Other channels follow the user's preferences:
  prefs = {"channels": ["email"], "quiet_hours": ["22:00", "07:00"], "muted_events": ["po.changed"]}
Mandatory events (approval.requested) ignore muted_events and quiet hours.
A notification row is unique per (event, user), so replays and retries never double-notify; a
channel that fails is recorded in channel_status and retried without re-creating the row.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Callable, Dict, List, Optional, Protocol

from services.people.events import DomainEvent
from services.people.models import EntityRef, Person
from services.people.resolver import RecipientResolver

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 5


class ChannelSkipped(Exception):
    """A channel cannot apply to this person (e.g. no address). Recorded, never retried."""


class NotificationRepo(Protocol):
    def insert(self, event: DomainEvent, user: str, reason: str, detail: str, title: str,
               body: str, link: str, priority: str) -> Optional[int]:
        """Returns the new id, or None when (event,user) already has a notification."""
    def lookup(self, event: DomainEvent, user: str) -> Optional[tuple]:
        """(id, channel_status dict) of the existing notification for (event,user)."""
    def set_channel_status(self, notification_id: int, channel: str, status: str) -> None: ...


class InMemoryNotificationRepo:
    def __init__(self):
        self.rows: List[dict] = []

    def insert(self, event, user, reason, detail, title, body, link, priority):
        if any(r["event_id"] == event.event_id and r["user"] == user for r in self.rows):
            return None
        row = dict(id=len(self.rows) + 1, event_id=event.event_id, tenant_id=event.tenant_id, user=user,
                   reason=reason, detail=detail, title=title, body=body, link=link,
                   priority=priority, read_at=None, channels={"in_app": "sent"})
        self.rows.append(row)
        return row["id"]

    def lookup(self, event, user):
        for r in self.rows:
            if r["event_id"] == event.event_id and r["user"] == user:
                return r["id"], dict(r["channels"])
        return None

    def set_channel_status(self, notification_id, channel, status):
        self.rows[notification_id - 1]["channels"][channel] = status


class PgNotificationRepo:
    def __init__(self, conn):
        self._conn = conn

    def insert(self, event, user, reason, detail, title, body, link, priority):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_notification (tenant_id,event_id,user_subject,reason,detail,title,body,"
                "link,priority,channel_status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'{\"in_app\":\"sent\"}'::jsonb)"
                " ON CONFLICT (event_id,user_subject) DO NOTHING RETURNING notification_id",
                (event.tenant_id, event.event_id, user, reason, detail, title, body, link, priority))
            r = cur.fetchone()
            return r[0] if r else None

    def lookup(self, event, user):
        with self._conn.cursor() as cur:
            cur.execute("SELECT notification_id, channel_status FROM proc.bp_notification "
                        "WHERE event_id=%s AND user_subject=%s", (event.event_id, user))
            r = cur.fetchone()
            return (r[0], dict(r[1] or {})) if r else None

    def set_channel_status(self, notification_id, channel, status):
        with self._conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_notification SET channel_status = channel_status || "
                        "jsonb_build_object(%s, %s::text) WHERE notification_id=%s",
                        (channel, status, notification_id))


def _in_quiet_hours(prefs: dict, now: datetime) -> bool:
    qh = prefs.get("quiet_hours")
    if not qh:
        return False
    start, end = (time.fromisoformat(x) for x in qh)
    t = now.timetz().replace(tzinfo=None)
    return (start <= t < end) if start <= end else (t >= start or t < end)


class NotificationService:
    def __init__(
        self,
        resolver: RecipientResolver,
        repo: NotificationRepo,
        entity_for: Callable[[DomainEvent], EntityRef],
        person_for: Callable[[str, str], Optional[Person]],
        prefs_for: Callable[[str, str], dict] = lambda tenant, subject: {},
        channels: Optional[Dict[str, Callable[[Person, dict], None]]] = None,
        link_for: Callable[[DomainEvent], str] = lambda e: f"/{e.entity_type}/{e.entity_id}",
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        on_created: Optional[Callable[[str, DomainEvent], None]] = None,
    ):
        self._resolver, self._repo, self._entity_for = resolver, repo, entity_for
        self._person_for, self._prefs_for = person_for, prefs_for
        self._channels, self._link_for, self._now = channels or {}, link_for, now
        self._on_created = on_created

    def handle(self, event: DomainEvent) -> int:
        """Create and dispatch notifications for one event; returns how many were newly created.
        Raises if a channel is down so the worker retries; rows already written stay (idempotent)."""
        actor = self._person_for(event.tenant_id, event.actor_id)
        if actor is None:
            logger.info("event %s: actor %s is not in bp_person, nobody notified", event.event_id, event.actor_id)
            return 0
        data = event.data or {}
        recipients = self._resolver.resolve(actor, event.type, self._entity_for(event),
                                            explicit=data.get("explicit") or (), mentioned=data.get("mentioned") or ())
        created, failed = 0, []
        for r in recipients:
            prefs = self._prefs_for(event.tenant_id, r.person.subject) or {}
            if not event.mandatory and event.type in prefs.get("muted_events", []):
                continue
            # The title is the event's one-line summary; the body is what the event says it is
            # (a message preview and what it is about), or the summary again when it says nothing.
            body = str(data.get("body") or event.summary)
            nid = self._repo.insert(event, r.person.subject, r.reason, r.detail, event.summary,
                                    body, self._link_for(event), event.priority)
            done_channels: dict = {}
            if nid is None:                 # replay or retry: only channels not yet sent run again
                found = self._repo.lookup(event, r.person.subject)
                if found is None:
                    continue
                nid, done_channels = found
            else:
                created += 1
                if self._on_created:
                    try:
                        self._on_created(r.person.subject, event)
                    except Exception:
                        logger.debug("live signal failed", exc_info=True)
            quiet = (not event.mandatory) and _in_quiet_hours(prefs, self._now())
            for ch in prefs.get("channels", []):
                send = self._channels.get(ch)
                if send is None or done_channels.get(ch) == "sent":
                    continue
                if quiet:
                    self._repo.set_channel_status(nid, ch, "deferred_quiet_hours")
                    continue
                try:
                    send(r.person, {"title": event.summary, "body": body, "detail": r.detail,
                                    "link": self._link_for(event)})
                    self._repo.set_channel_status(nid, ch, "sent")
                except ChannelSkipped as skip:
                    self._repo.set_channel_status(nid, ch, f"skipped_{skip}")
                except Exception as exc:
                    logger.warning("channel %s failed for %s: %s", ch, r.person.subject, exc)
                    self._repo.set_channel_status(nid, ch, "failed")
                    failed.append(ch)
        if failed:
            raise RuntimeError(f"channels failed: {sorted(set(failed))}")
        return created


class Outbox(Protocol):
    def claim(self, limit: int) -> List[DomainEvent]: ...
    def done(self, event_id: str) -> None: ...
    def failed(self, event_id: str, error: str, give_up: bool) -> None: ...


def drain(outbox: Outbox, service: NotificationService, limit: int = 50) -> Dict[str, int]:
    """Process pending events once. A failing event is retried on later drains up to MAX_ATTEMPTS."""
    stats = {"done": 0, "retry": 0, "gave_up": 0}
    for ev in outbox.claim(limit):
        try:
            service.handle(ev)
            outbox.done(ev.event_id)
            stats["done"] += 1
        except Exception as exc:
            give_up = ev.attempts + 1 >= MAX_ATTEMPTS
            outbox.failed(ev.event_id, str(exc)[:500], give_up)
            stats["gave_up" if give_up else "retry"] += 1
            logger.error("event %s (%s) failed attempt %s: %s", ev.event_id, ev.type, ev.attempts + 1, exc)
    return stats


class PgOutbox:
    """Run ONE worker. claim() takes no lock, so two workers could both process an event; that is
    safe (notifications are unique per event+user) but wasteful, and channels could double-send."""

    def __init__(self, conn):
        self._conn = conn

    def claim(self, limit):
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT event_id,event_type,tenant_id,actor_id,entity_type,entity_id,summary,data,occurred_at,attempts"
                " FROM proc.bp_domain_event WHERE processed_at IS NULL AND attempts < %s"
                " ORDER BY occurred_at LIMIT %s", (MAX_ATTEMPTS, limit))
            return [DomainEvent(type=r[1], tenant_id=r[2], actor_id=r[3], entity_type=r[4], entity_id=r[5],
                                summary=r[6], data=r[7] or {}, event_id=r[0], occurred_at=r[8], attempts=r[9])
                    for r in cur.fetchall()]

    def done(self, event_id):
        with self._conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_domain_event SET processed_at=now(), last_error=NULL WHERE event_id=%s", (event_id,))

    def failed(self, event_id, error, give_up):
        with self._conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_domain_event SET attempts=attempts+1, last_error=%s WHERE event_id=%s",
                        (error, event_id))
