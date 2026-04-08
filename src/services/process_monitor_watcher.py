"""Real-time watcher for proc.process_monitor table.

Listens for PostgreSQL NOTIFY events when document uploads complete
(status = 'Completed') and dispatches data extraction concurrently.
Falls back to polling every 60s for resilience.
"""

from __future__ import annotations

import logging
import select
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extensions

logger = logging.getLogger(__name__)

LISTEN_CHANNEL = "process_monitor_ready"
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_MAX_WORKERS = 4
MAX_BACKOFF = 60.0


class ProcessMonitorWatcher:
    """Watches proc.process_monitor for completed uploads and triggers extraction."""

    def __init__(
        self,
        agent_nick,
        *,
        orchestrator=None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self._agent_nick = agent_nick
        self._orchestrator = orchestrator
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._listen_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="process-monitor-extract",
        )
        self._processing_ids: set[int] = set()
        self._processing_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _get_connection(self, autocommit: bool = True) -> psycopg2.extensions.connection:
        """Create a fresh psycopg2 connection using agent_nick settings."""
        settings = self._agent_nick.settings
        conn = psycopg2.connect(
            host=settings.db_host,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            port=settings.db_port,
        )
        conn.autocommit = autocommit
        return conn

    # ------------------------------------------------------------------
    # Trigger management
    # ------------------------------------------------------------------
    def _ensure_trigger(self) -> None:
        """Ensure the process_monitor table has proper schema and trigger."""
        schema_sql = """
        -- Ensure id column has a sequence default and is the primary key
        ALTER TABLE proc.process_monitor
            ALTER COLUMN id SET DEFAULT nextval('proc.process_monitor_id_seq');

        -- Backfill any NULL ids
        UPDATE proc.process_monitor
        SET id = nextval('proc.process_monitor_id_seq')
        WHERE id IS NULL;

        -- Ensure NOT NULL
        ALTER TABLE proc.process_monitor
            ALTER COLUMN id SET NOT NULL;

        -- Add primary key if missing
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'proc.process_monitor'::regclass
                AND contype = 'p'
            ) THEN
                ALTER TABLE proc.process_monitor ADD PRIMARY KEY (id);
            END IF;
        END;
        $$;
        """
        trigger_sql = """
        CREATE OR REPLACE FUNCTION proc.notify_process_monitor_ready()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.status = 'Completed' AND NEW.id IS NOT NULL THEN
                PERFORM pg_notify('process_monitor_ready', NEW.id::text);
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger WHERE tgname = 'trg_process_monitor_ready'
            ) THEN
                CREATE TRIGGER trg_process_monitor_ready
                    AFTER INSERT OR UPDATE ON proc.process_monitor
                    FOR EACH ROW
                    EXECUTE FUNCTION proc.notify_process_monitor_ready();
            END IF;
        END;
        $$;
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                    cur.execute(trigger_sql)
                logger.info("process_monitor schema and trigger ensured")
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to ensure process_monitor schema/trigger")

    # ------------------------------------------------------------------
    # Claim + process
    # ------------------------------------------------------------------
    def _claim_record(self, conn, record_id: int) -> Optional[Dict[str, Any]]:
        """Atomically claim a record by transitioning Completed -> Extracting.

        Returns the record dict if successfully claimed, None otherwise.
        """
        with self._processing_lock:
            if record_id in self._processing_ids:
                return None
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE proc.process_monitor
                    SET status = 'Extracting', start_ts = %s
                    WHERE id = %s AND status = 'Completed'
                    RETURNING id, process_name, type, status, file_path,
                              start_ts, created_date, created_by,
                              lastmodified_date, end_ts, category,
                              document_type, user_id, total_count
                    """,
                    (datetime.now(timezone.utc), record_id),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                columns = [desc.name for desc in cur.description]
                record = dict(zip(columns, row))
                with self._processing_lock:
                    self._processing_ids.add(record_id)
                return record
        except Exception:
            logger.exception("Failed to claim record %s", record_id)
            return None

    def _mark_extracted(self, record_id: int) -> None:
        """Mark a record as successfully extracted."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.process_monitor
                        SET status = 'Extracted',
                            end_ts = %s,
                            lastmodified_date = %s
                        WHERE id = %s
                        """,
                        (
                            datetime.now(timezone.utc),
                            datetime.now(timezone.utc),
                            record_id,
                        ),
                    )
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to mark record %s as Extracted", record_id)
        finally:
            with self._processing_lock:
                self._processing_ids.discard(record_id)

    def _mark_failed(self, record_id: int, error: str) -> None:
        """Mark a record as failed extraction."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.process_monitor
                        SET status = 'Extraction_Failed',
                            end_ts = %s,
                            lastmodified_date = %s
                        WHERE id = %s
                        """,
                        (
                            datetime.now(timezone.utc),
                            datetime.now(timezone.utc),
                            record_id,
                        ),
                    )
            finally:
                conn.close()
        except Exception:
            logger.exception(
                "Failed to mark record %s as Extraction_Failed", record_id
            )
        finally:
            with self._processing_lock:
                self._processing_ids.discard(record_id)

    def _process_record(self, record: Dict[str, Any]) -> None:
        """Run extraction for a claimed record."""
        record_id = record["id"]
        file_path = record.get("file_path", "")
        category = record.get("category", "")
        document_type = record.get("document_type", "")
        user_id = record.get("user_id")
        logger.info(
            "Starting extraction for record %s: file_path=%s category=%s",
            record_id,
            file_path,
            category,
        )
        try:
            # Try direct extraction first (schema-driven, LLM-powered)
            try:
                from services.direct_extraction_service import DirectExtractionService
                agent_nick = self._agent_nick
                if agent_nick and hasattr(agent_nick, "settings") and hasattr(agent_nick.settings, "s3_bucket_name"):
                    service = DirectExtractionService(agent_nick)
                    result = service.extract_and_persist(
                        file_path,
                        category,
                        user_id=str(user_id) if user_id is not None else None,
                    )
                    status = str(result.get("status", "error")).lower()
                    if status == "success":
                        self._mark_extracted(record_id)
                        logger.info(
                            "Direct extraction completed for record %s: %s pk=%s header=%s lines=%s",
                            record_id, result.get("doc_type"), result.get("pk"),
                            result.get("header_fields"), result.get("line_items"),
                        )
                        return
                    logger.warning(
                        "Direct extraction failed for record %s: %s — falling back to orchestrator",
                        record_id, result.get("error"),
                    )
            except Exception as direct_exc:
                logger.warning(
                    "Direct extraction unavailable for record %s: %s — using orchestrator",
                    record_id, direct_exc,
                )

            # Fallback to orchestrator pipeline
            orchestrator = self._orchestrator
            if orchestrator is None:
                raise RuntimeError("Orchestrator not available")
            result = orchestrator.execute_extraction_flow(
                s3_object_key=file_path,
                category=category,
                document_type=document_type,
                user_id=str(user_id) if user_id is not None else None,
            )
            status = "error"
            if isinstance(result, dict):
                status = str(result.get("status", "error")).lower()
            if status in ("blocked", "error", "failed"):
                raise RuntimeError(f"Extraction {status}: {result.get('reason', result.get('error', 'unknown'))}")
            self._mark_extracted(record_id)
            logger.info("Extraction completed for record %s", record_id)
        except Exception as exc:
            logger.exception("Extraction failed for record %s", record_id)
            self._mark_failed(record_id, str(exc))

    def _dispatch(self, record: Dict[str, Any]) -> None:
        """Submit a record for extraction on the thread pool."""
        self._executor.submit(self._process_record, record)

    # ------------------------------------------------------------------
    # Listener thread (LISTEN/NOTIFY)
    # ------------------------------------------------------------------
    def _listen_loop(self) -> None:
        """Dedicated thread: holds a PG connection with LISTEN, dispatches on NOTIFY."""
        backoff = 2.0
        while not self._stop_event.is_set():
            conn = None
            try:
                conn = self._get_connection(autocommit=True)
                conn.set_isolation_level(
                    psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
                )
                with conn.cursor() as cur:
                    cur.execute(f"LISTEN {LISTEN_CHANNEL}")
                logger.info("LISTEN connection established on channel '%s'", LISTEN_CHANNEL)
                backoff = 2.0  # reset on successful connect

                while not self._stop_event.is_set():
                    if select.select([conn], [], [], 1.0) == ([], [], []):
                        continue  # timeout, check stop_event
                    conn.poll()
                    while conn.notifies:
                        notify = conn.notifies.pop(0)
                        try:
                            record_id = int(notify.payload)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Invalid NOTIFY payload: %s", notify.payload
                            )
                            continue
                        logger.debug("NOTIFY received for record %s", record_id)
                        claim_conn = self._get_connection()
                        try:
                            record = self._claim_record(claim_conn, record_id)
                        finally:
                            claim_conn.close()
                        if record:
                            self._dispatch(record)

            except Exception:
                if not self._stop_event.is_set():
                    logger.exception(
                        "LISTEN connection error, reconnecting in %.1fs", backoff
                    )
                    self._stop_event.wait(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Poller thread (fallback)
    # ------------------------------------------------------------------
    def _poll_loop(self) -> None:
        """Fallback poller: sweeps for Completed records every poll_interval seconds."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self._poll_interval)
            if self._stop_event.is_set():
                break
            try:
                self._sweep_completed()
            except Exception:
                logger.exception("Poll sweep failed")

    def _sweep_completed(self) -> None:
        """Query for Completed records and stale Extracting records, then dispatch."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Recover stale Extracting records (stuck > 10 minutes)
                cur.execute(
                    """
                    UPDATE proc.process_monitor
                    SET status = 'Completed', start_ts = NULL
                    WHERE status = 'Extracting'
                      AND start_ts < NOW() - INTERVAL '30 minutes'
                    RETURNING id
                    """
                )
                stale = cur.fetchall()
                if stale:
                    logger.warning(
                        "Recovered %d stale Extracting records: %s",
                        len(stale), [r[0] for r in stale],
                    )

                cur.execute(
                    """
                    SELECT id FROM proc.process_monitor
                    WHERE status = 'Completed'
                    ORDER BY created_date ASC
                    """
                )
                rows = cur.fetchall()
            for (record_id,) in rows:
                claim_conn = self._get_connection()
                try:
                    record = self._claim_record(claim_conn, record_id)
                finally:
                    claim_conn.close()
                if record:
                    self._dispatch(record)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start listener and poller threads."""
        if self._listen_thread and self._listen_thread.is_alive():
            return
        self._stop_event.clear()
        self._ensure_trigger()

        # Initial sweep for any records that arrived before we started
        try:
            self._sweep_completed()
        except Exception:
            logger.exception("Initial sweep failed")

        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            name="process-monitor-listener",
            daemon=True,
        )
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="process-monitor-poller",
            daemon=True,
        )
        self._listen_thread.start()
        self._poll_thread.start()
        logger.info("ProcessMonitorWatcher started")

    def stop(self) -> None:
        """Stop all threads and wait for in-flight extractions."""
        self._stop_event.set()
        for thread in (self._listen_thread, self._poll_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5)
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("ProcessMonitorWatcher stopped")

    def update_orchestrator(self, orchestrator) -> None:
        """Update the orchestrator reference (called when scheduler refreshes)."""
        self._orchestrator = orchestrator
