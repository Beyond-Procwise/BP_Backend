"""proc.bp_egress_event — one row per outbound call.

The audit asked: "can you reconstruct, for a given date, precisely what left the
boundary and under which policy version?" The answer was no for 21 of 23 paths,
because nothing recorded that the call had happened. ``egress._record`` fixed
that for the log; this puts it somewhere queryable.

WHY IT IS BUFFERED

``services.db.get_conn()`` opens a fresh psycopg2 connection per call — there is
no pool. A synchronous INSERT per outbound call would mean a TCP connect to
Postgres for every model inference, and the extraction pipeline makes several
per document. So events go onto a bounded in-memory queue and a single
background thread flushes them in batches over one connection.

The cost of that choice, stated rather than discovered later:

  * Events sitting in the queue when the process is killed -9 are lost. An
    atexit hook flushes on clean shutdown, which covers restarts and deploys but
    not a hard kill or an OOM.
  * The queue is bounded. If the writer cannot keep up, events are DROPPED —
    and the drop is COUNTED and written as its own row, because an audit log
    that silently loses entries is worse than one that admits to a gap. A
    reader can see "43 events were dropped here" rather than seeing nothing.

WHAT THE ROWS DO NOT YET CARRY, and why the columns exist anyway

  tenant_id       always 'default'. There is no tenant dimension in this
                  product (see api/auth.py). The column exists so the rows do
                  not need rewriting when there is one.
  policy_version  always NULL. No policy model keyed on
                  (destination, purpose, classification) exists to have a
                  version. egress._evaluate is the seam that will supply it.
  classification  always empty. There is no data-classification registry.

Those three being empty is the honest current state of the controls, not an
oversight in this module. A row that claimed a policy version it did not have
would be worse than a NULL.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DDL = """
CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_egress_event (
    event_id        BIGSERIAL PRIMARY KEY,
    occurred_at     TIMESTAMPTZ NOT NULL,
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    purpose         TEXT NOT NULL,
    destination     TEXT NOT NULL,
    method          TEXT,
    outcome         TEXT NOT NULL,
    detail          TEXT,
    -- SHA-256 of the exact request body, never the body itself. Lets a reader
    -- prove that a given payload was or was not sent without the log becoming
    -- a second copy of the data it is auditing.
    payload_sha256  TEXT,
    payload_bytes   INTEGER,
    -- NULL until a destination policy model exists. See the module docstring.
    policy_version  TEXT,
    classification  TEXT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_bp_egress_event_occurred_at
    ON proc.bp_egress_event (occurred_at DESC);

CREATE INDEX IF NOT EXISTS ix_bp_egress_event_purpose_occurred
    ON proc.bp_egress_event (purpose, occurred_at DESC);

CREATE INDEX IF NOT EXISTS ix_bp_egress_event_destination_occurred
    ON proc.bp_egress_event (destination, occurred_at DESC);
"""

_INSERT = """
INSERT INTO proc.bp_egress_event
    (occurred_at, tenant_id, purpose, destination, method, outcome, detail,
     payload_sha256, payload_bytes, policy_version, classification)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Bounded so a writer that cannot keep up costs memory that is capped rather
# than unbounded. 10k events is a few MB and several minutes of heavy traffic.
_MAX_QUEUE = 10_000
_BATCH = 200
_FLUSH_SECONDS = 2.0

_queue: "queue.Queue[dict]" = queue.Queue(maxsize=_MAX_QUEUE)
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_dropped = 0
_dropped_lock = threading.Lock()
_enabled = True


def payload_digest(body: Any) -> tuple[Optional[str], Optional[int]]:
    """(sha256, byte length) for a request body, or (None, None) if there is none.

    The hash is of the bytes as sent. ``json=`` bodies are serialised with sorted
    keys so the same payload hashes the same way regardless of dict ordering —
    otherwise two identical requests would produce two different digests and the
    hash would prove nothing.
    """
    if body is None:
        return None, None
    try:
        if isinstance(body, (bytes, bytearray)):
            raw = bytes(body)
        elif isinstance(body, str):
            raw = body.encode("utf-8")
        else:
            raw = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    except Exception:  # noqa: BLE001 - a body we cannot serialise is not fatal
        return None, None
    return hashlib.sha256(raw).hexdigest(), len(raw)


def record(*, purpose: str, destination: str, method: str, outcome: str,
           detail: str = "", payload_sha256: Optional[str] = None,
           payload_bytes: Optional[int] = None) -> None:
    """Queue one egress event. Never blocks, never raises.

    Called from the request path, so it must cost close to nothing and must not
    be able to fail a call it is only observing.
    """
    if not _enabled:
        return
    event = {
        "occurred_at": datetime.now(timezone.utc),
        "purpose": purpose,
        "destination": destination,
        "method": method,
        "outcome": outcome,
        "detail": detail or None,
        "payload_sha256": payload_sha256,
        "payload_bytes": payload_bytes,
    }
    try:
        _queue.put_nowait(event)
    except queue.Full:
        _count_drop()
        return
    except Exception:  # noqa: BLE001 - see below
        _count_drop()
        return

    try:
        _ensure_worker()
    except Exception:  # noqa: BLE001
        # Starting the writer can fail — thread limits, an interpreter already
        # shutting down. The event is queued either way and the next successful
        # record() or the atexit flush will carry it. What must NOT happen is
        # this propagating: record() is called from inside the request path, and
        # an audit writer that can raise into the call it is observing would
        # turn a logging problem into a failed outbound request.
        logger.debug("egress log writer could not be started", exc_info=True)


def _count_drop() -> None:
    global _dropped
    with _dropped_lock:
        _dropped += 1


def _take_drops() -> int:
    global _dropped
    with _dropped_lock:
        n, _dropped = _dropped, 0
    return n


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(
            target=_run, name="egress-log-writer", daemon=True
        )
        _worker.start()


def _drain(limit: int = _BATCH) -> list[dict]:
    batch: list[dict] = []
    while len(batch) < limit:
        try:
            batch.append(_queue.get_nowait())
        except queue.Empty:
            break
    return batch


def _row(event: dict) -> tuple:
    return (
        event["occurred_at"], "default", event["purpose"], event["destination"],
        event["method"], event["outcome"], event["detail"],
        event["payload_sha256"], event["payload_bytes"],
        None,      # policy_version — no policy model yet
        [],        # classification — no registry yet
    )


def flush(batch: Optional[list[dict]] = None) -> int:
    """Write queued events. Returns rows written. Never raises.

    A failure here must not propagate: this observes outbound calls, and an
    audit writer that can break the thing it audits is a worse problem than a
    gap in the audit.
    """
    events = _drain() if batch is None else batch
    dropped = _take_drops()
    if dropped:
        # The gap is recorded as a row of its own. An audit log that silently
        # loses entries is worse than one that admits to a gap: a reader can act
        # on "43 events were dropped", and cannot act on their absence.
        events.append({
            "occurred_at": datetime.now(timezone.utc),
            "purpose": "audit", "destination": "-", "method": "-",
            "outcome": "events_dropped",
            "detail": f"{dropped} event(s) dropped: the writer could not keep up",
            "payload_sha256": None, "payload_bytes": None,
        })
    if not events:
        return 0
    try:
        from src.services.db import get_conn
        with get_conn() as conn:
            cur = conn.cursor()
            cur.executemany(_INSERT, [_row(e) for e in events])
            if not getattr(conn, "autocommit", False):
                conn.commit()
        return len(events)
    except Exception:
        logger.debug("egress log flush failed; %d event(s) lost",
                     len(events), exc_info=True)
        return 0


def _run() -> None:
    while True:
        try:
            # Block for the first event so an idle process does no work, then
            # take whatever else has accumulated in one batch.
            first = _queue.get(timeout=_FLUSH_SECONDS)
        except queue.Empty:
            if _take_drops():
                flush([])
            continue
        except Exception:  # pragma: no cover - defensive
            return
        batch = [first] + _drain(_BATCH - 1)
        flush(batch)


def ensure_schema() -> None:
    """Create the table. Safe to call repeatedly."""
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(DDL)
        if not getattr(conn, "autocommit", False):
            conn.commit()


def _flush_at_exit() -> None:
    """Write whatever is queued on a clean shutdown.

    Covers restarts and deploys. It does not cover a hard kill or an OOM, and
    the module docstring says so rather than implying the queue is durable.
    """
    remaining = _drain(_MAX_QUEUE)
    if remaining or _dropped:
        flush(remaining)


atexit.register(_flush_at_exit)
