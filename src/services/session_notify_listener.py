"""Background LISTEN thread that forwards PostgreSQL session_status
notifications to connected WebSocket clients.

Architecture
------------
PostgreSQL fires::

    pg_notify('session_status', <json>)

…inside the trigger function proc.fn_try_resolve_session() the moment every
document in a session reaches a terminal outcome.

This module runs a daemon thread that holds a persistent psycopg2 connection
with ``LISTEN session_status`` active.  When a notification arrives the thread
parses the JSON payload and bridges into the asyncio event loop via
``asyncio.run_coroutine_threadsafe`` to call
``ws_manager.broadcast_to_session``.

Resilience
----------
* Any exception in the LISTEN loop triggers a reconnect after
  ``RECONNECT_DELAY`` seconds (configurable via
  ``SESSION_NOTIFY_RECONNECT_DELAY`` env var).
* The stop event is checked every 2 seconds via ``select`` timeout, so shutdown
  is prompt.
* All exceptions are logged; none are re-raised.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import threading
import time
from typing import Optional

import psycopg2
import psycopg2.extensions

from config.settings import Settings

log = logging.getLogger(__name__)

LISTEN_CHANNEL = "session_status"
RECONNECT_DELAY = int(os.getenv("SESSION_NOTIFY_RECONNECT_DELAY", "10"))
_SELECT_TIMEOUT = 2.0  # seconds — controls how quickly stop() is noticed


class SessionNotifyListener:
    """Long-running daemon thread.  Owns a single psycopg2 LISTEN connection.

    Pass the running asyncio event loop so the thread can safely invoke async
    ws_manager methods without creating its own loop.
    """

    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self._loop = event_loop
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon thread.  Idempotent — safe to call multiple times."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="session-notify-listener",
            daemon=True,
        )
        self._thread.start()
        log.info("SessionNotifyListener started (channel='%s')", LISTEN_CHANNEL)

    def stop(self) -> None:
        """Signal the thread to exit.  Returns immediately; thread exits within ~2s."""
        self._stop.set()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Outer loop: reconnect on crash with back-off."""
        while not self._stop.is_set():
            try:
                self._listen_loop()
            except Exception as exc:
                if self._stop.is_set():
                    return
                log.warning(
                    "SessionNotifyListener loop crashed — reconnecting in %ds: %s",
                    RECONNECT_DELAY, exc,
                )
                if self._stop.wait(RECONNECT_DELAY):
                    return

    def _listen_loop(self) -> None:
        """Connect, LISTEN, dispatch notifications until stop() is called."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(f"LISTEN {LISTEN_CHANNEL};")
            log.info(
                "SessionNotifyListener: LISTEN active on channel '%s'",
                LISTEN_CHANNEL,
            )
            while not self._stop.is_set():
                # select() blocks up to _SELECT_TIMEOUT seconds; gives the stop
                # event a chance to be noticed without busy-spinning.
                readable, _, _ = select.select([conn], [], [], _SELECT_TIMEOUT)
                if not readable:
                    continue
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    self._dispatch(notify.payload)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _connect(self) -> psycopg2.extensions.connection:
        s = Settings()
        conn = psycopg2.connect(
            host=s.db_host,
            dbname=s.db_name,
            user=s.db_user,
            password=s.db_password,
            port=s.db_port,
        )
        conn.autocommit = True
        return conn

    def _dispatch(self, payload_str: str) -> None:
        """Parse the JSON payload and schedule a WebSocket broadcast."""
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            log.warning(
                "SessionNotifyListener: malformed JSON payload — skipped: %r",
                payload_str,
            )
            return

        session_id: Optional[str] = payload.get("session_id")
        if not session_id:
            log.warning(
                "SessionNotifyListener: payload missing session_id — skipped: %r",
                payload,
            )
            return

        log.info(
            "Session resolved via NOTIFY  session_id=%s  action_status=%s  "
            "total=%s  target=%s  discrepancy=%s  failed=%s",
            session_id,
            payload.get("action_status"),
            payload.get("total"),
            payload.get("target"),
            payload.get("discrepancy"),
            payload.get("failed"),
        )

        # Link the upload to its deal BEFORE telling the client it is ready.
        # The terminal frame is the report page's cue to stop waiting and render,
        # so sending it while the documents are still sitting in _stg is what
        # produced an empty "Document Analysis Report" for up to 11 minutes.
        # Run it off the LISTEN thread so notifications keep flowing.
        threading.Thread(
            target=self._link_then_broadcast,
            args=(session_id, payload),
            name=f"session-finalise-{session_id}",
            daemon=True,
        ).start()

    def _link_then_broadcast(self, session_id: str, payload: dict) -> None:
        """Link, freeze, then broadcast — and broadcast whatever happens.

        Fail-open is deliberate: a linking or freezing error must never strand
        the page on a spinner. The client is told the session resolved either
        way; a failure shows up as a report with no documents or no findings,
        which is what it would have shown before this existed. A freeze that
        did not happen is picked up by the scheduled sweep.
        """
        try:
            self._link_session(session_id)
        except Exception:
            log.exception(
                "Fast deal-linking failed for session %s — broadcasting anyway; "
                "the scheduled sweep will pick it up",
                session_id,
            )
        try:
            self._freeze_analysis(session_id)
        except Exception:
            log.exception(
                "Freezing the analysis for session %s failed — broadcasting "
                "anyway; the scheduled sweep will retry it",
                session_id,
            )
        self._broadcast(session_id, payload)

    def _freeze_analysis(self, session_id: str) -> None:
        """Turn the running analysis for this session into a frozen record.

        Runs after linking (the deals only exist once linking has happened)
        and before broadcasting — see the fail-open contract on
        ``_link_then_broadcast``.
        """
        from src.services import analysis_findings, analysis_store  # noqa: PLC0415

        deal_ids = analysis_store.deal_ids_for_session(session_id)
        file_paths = analysis_store.file_paths_for_session(session_id)
        findings = analysis_findings.capture(deal_ids, file_paths=file_paths)
        value_found, currency = analysis_findings.headline(findings)
        analysis_store.freeze(
            session_id,
            findings=findings,
            document_count=analysis_store.document_count_for_session(session_id),
            value_found=value_found,
            currency=currency,
        )

    def _link_session(self, session_id: str) -> None:
        """Run the sub-second linking passes for the documents that just landed.

        Disable with SESSION_FAST_LINK_ENABLED=0 to fall back to the old
        behaviour (linking only on the 15-minute scheduled sweep).
        """
        if os.getenv("SESSION_FAST_LINK_ENABLED", "1").strip() in ("0", "false", "False"):
            return
        from src.services.deal_assignment_service import assign_deals_fast  # noqa: PLC0415

        started = time.monotonic()
        result = assign_deals_fast()
        log.info(
            "Session %s linked in %.2fs before broadcast: %s",
            session_id, time.monotonic() - started, result,
        )

    def _broadcast(self, session_id: str, payload: dict) -> None:
        # Import here to avoid circular imports at module load time.
        from src.services.ws_manager import ws_manager  # noqa: PLC0415

        # Bridge: this method runs in a plain thread; ws_manager is async.
        # run_coroutine_threadsafe schedules the coroutine on the FastAPI event
        # loop and returns a concurrent.futures.Future (we don't await it).
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast_to_session(session_id, payload),
            self._loop,
        )
