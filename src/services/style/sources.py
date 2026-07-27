"""Where exemplars come from.

The most important structural decision in this build is that compilation never knows.
It asks an ``ExemplarSource`` for raw emails and gets them; whether they were pasted into
a form, read from a bound Microsoft 365 mailbox or pulled over IMAP is the source's
business alone.

Only ``PastedExemplarSource`` exists today. Mode C sources arrive later as additional
implementations of this protocol, and when they do, nothing in ``compiler.py`` should
need to change. If it does, the seam was drawn in the wrong place.

**No source may send mail.** These adapters read. The platform's SES dispatch path is a
separate, deliberately-retained subsystem (see the Phase -1 inventory, escalation E1) and
must never be reachable from here — a test asserts no send-shaped method exists on any
implementation of this protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, runtime_checkable

from services.db import get_conn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawExemplar:
    """One email, as it arrived, before redaction.

    Short-lived by design: instances exist between a source and the redactor and are not
    persisted anywhere in this form.
    """

    body: str
    subject: Optional[str] = None
    intent: Optional[str] = None
    source_ref: Optional[str] = None
    # Set by staging-backed sources so the compiler can purge exactly what it consumed.
    staging_id: Optional[int] = None
    # Set by mailbox-backed sources. Under Mode C2 this is the ONLY identifier a draft
    # can cite, because the body itself is never written anywhere.
    message_id: Optional[str] = None


@runtime_checkable
class ExemplarSource(Protocol):
    """Supplies raw emails for a user, optionally narrowed to one intent."""

    def fetch(
        self, user_ref: str, intent: str | None = None
    ) -> List[RawExemplar]:  # pragma: no cover - protocol
        ...


class PastedExemplarSource:
    """Reads emails a user pasted into the platform, from ``bp_style_ingest_staging``.

    Staging is a queue, not a store: the compiler deletes what it consumes, and the TTL
    sweep deletes whatever it does not. This class only reads.
    """

    def __init__(self, conn: Optional[Any] = None) -> None:
        self._conn = conn

    def _with_conn(self, fn):
        if self._conn is not None:
            return fn(self._conn)
        with get_conn() as conn:
            return fn(conn)

    def fetch(self, user_ref: str, intent: str | None = None) -> List[RawExemplar]:
        """Every un-purged pasted email for this user, oldest first.

        ``intent`` narrows to a single communication type. Left as None — the usual case,
        since the user-level profile is the primary artifact — everything is returned
        regardless of how each email was tagged.
        """

        def _run(conn):
            cur = conn.cursor()
            if intent:
                cur.execute(
                    "SELECT ingest_id, subject, body, intent, source_ref "
                    "FROM proc.bp_style_ingest_staging "
                    "WHERE user_ref = %s AND intent = %s ORDER BY created_at, ingest_id",
                    (user_ref, intent),
                )
            else:
                cur.execute(
                    "SELECT ingest_id, subject, body, intent, source_ref "
                    "FROM proc.bp_style_ingest_staging "
                    "WHERE user_ref = %s ORDER BY created_at, ingest_id",
                    (user_ref,),
                )
            rows = cur.fetchall()
            cur.close()
            return rows

        rows = self._with_conn(_run)
        return [
            RawExemplar(
                staging_id=row[0],
                subject=row[1],
                body=row[2],
                intent=row[3],
                source_ref=row[4],
            )
            for row in rows
            if row[2]
        ]

    # -- writes into the queue ----------------------------------------------------

    def stage(
        self,
        *,
        user_ref: str,
        batch_id: str,
        submitted_by: str,
        body: str,
        subject: Optional[str] = None,
        intent: Optional[str] = None,
        source_ref: Optional[str] = None,
    ) -> int:
        """Put one pasted email on the queue. Returns its staging id."""

        if not body or not body.strip():
            raise ValueError("cannot stage an empty email")

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO proc.bp_style_ingest_staging "
                "(user_ref, batch_id, intent, subject, body, source_ref, submitted_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING ingest_id",
                (user_ref, batch_id, intent, subject, body, source_ref, submitted_by),
            )
            ingest_id = cur.fetchone()[0]
            cur.close()
            if self._conn is None and hasattr(conn, "commit"):
                conn.commit()
            return ingest_id

        return self._with_conn(_run)
