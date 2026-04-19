"""Real-time watcher for proc.process_monitor table.

Listens for PostgreSQL NOTIFY events when document uploads complete
(status = 'Completed') and dispatches data extraction concurrently.
Falls back to polling every 60s for resilience.
"""

from __future__ import annotations

import logging
import os
import select
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extensions

from services.agent_nick_orchestrator import AgentNickOrchestrator

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
            IF NEW.status IN ('Completed', 'Running') AND NEW.id IS NOT NULL THEN
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
        Performs an early duplicate check before claiming to reduce
        unnecessary processing and log noise.
        """
        with self._processing_lock:
            if record_id in self._processing_ids:
                return None
        try:
            with conn.cursor() as cur:
                # Early duplicate check — skip before claiming to reduce noise
                cur.execute(
                    "SELECT file_path, category FROM proc.process_monitor WHERE id = %s",
                    (record_id,),
                )
                pre_row = cur.fetchone()
                if pre_row and pre_row[0]:
                    file_path, category = pre_row
                    cur.execute(
                        "SELECT id FROM proc.process_monitor "
                        "WHERE file_path = %s AND status IN ('Extracted', 'Extracting') AND id != %s "
                        "LIMIT 1",
                        (file_path, record_id),
                    )
                    dup = cur.fetchone()
                    if dup and not self._data_needs_reextraction(cur, file_path, category or ""):
                        logger.debug(
                            "Early dedup: record %s is duplicate of %s — skipping claim",
                            record_id, dup[0],
                        )
                        # Mark as extracted without processing
                        cur.execute(
                            "UPDATE proc.process_monitor SET status = 'Extracted', "
                            "end_ts = %s, lastmodified_date = %s WHERE id = %s AND status IN ('Completed', 'Running')",
                            (datetime.now(timezone.utc), datetime.now(timezone.utc), record_id),
                        )
                        return None

                cur.execute(
                    """
                    UPDATE proc.process_monitor
                    SET status = 'Extracting', start_ts = %s
                    WHERE id = %s AND status IN ('Completed', 'Running')
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

    @staticmethod
    def _data_needs_reextraction(cur, file_path: str, category: str) -> bool:
        """Check if extracted data is missing from the target bp_ table.

        Returns True when the process_monitor says 'Extracted' but the
        actual business data was deleted (e.g. during cleanup), meaning
        the document must be re-extracted.
        """
        import re as _re

        fname = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
        cat = (category or "").lower()

        # Extract expected PK from filename
        pk_val = None
        table = None
        pk_col = None
        if cat == "po":
            m = _re.search(r"PO\s*(\d{4,})", fname)
            if m:
                pk_val, table, pk_col = m.group(1), "proc.bp_purchase_order", "po_id"
        elif cat == "invoice":
            m = _re.search(r"(INV[\-]?\s*[\w\-]+)", fname, _re.I)
            if m:
                pk_val = _re.sub(r"\s+", "", m.group(1))
                table, pk_col = "proc.bp_invoice", "invoice_id"
        elif cat in ("quote", "quotes"):
            # Try QUT first, then QTE, then bare numeric
            for pat in (r"QUT[\-\s]*([\d][\d\-]{2,})", r"QTE[\-\s]*([\d][\d\-]{2,})", r"(\d{5,})"):
                m = _re.search(pat, fname, _re.I)
                if m:
                    pk_val, table, pk_col = _re.sub(r"\s+", "", m.group(1)), "proc.bp_quote", "quote_id"
                    break

        if not pk_val or not table:
            return False  # can't determine — assume no re-extraction needed

        try:
            cur.execute(f"SELECT 1 FROM {table} WHERE {pk_col} = %s LIMIT 1", (pk_val,))
            return cur.fetchone() is None  # True = data missing, needs re-extraction
        except Exception:
            return False

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
        """Dispatch to AgentNick (primary agent) for document processing."""
        record_id = record["id"]
        file_path = record.get("file_path", "")
        category = record.get("category", "")
        user_id = record.get("user_id")
        logger.info(
            "Starting extraction for record %s: file_path=%s category=%s",
            record_id,
            file_path,
            category,
        )

        # Check for duplicate file_path (same document already extracted)
        # Only skip if the data actually exists in the target bp_ table,
        # not just based on process_monitor status (which can be stale
        # after DB cleanup).
        if file_path:
            try:
                conn = self._get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT id FROM proc.process_monitor "
                            "WHERE file_path = %s AND status IN ('Extracted', 'Extracting') AND id != %s "
                            "LIMIT 1",
                            (file_path, record_id),
                        )
                        existing = cur.fetchone()
                        # Verify data actually exists in the target table
                        if existing and not self._data_needs_reextraction(
                            cur, file_path, category
                        ):
                            pass  # genuine duplicate — proceed to skip
                        else:
                            existing = None  # stale marker — allow re-extraction
                        if existing:
                            logger.info(
                                "Duplicate detected: record %s has same file_path as "
                                "already-extracted record %s — logging discrepancy",
                                record_id, existing[0],
                            )
                            # Log duplicate to discrepancy table
                            cur.execute(
                                "INSERT INTO proc.bp_discrepancy_data "
                                "(doc_type, record_id, field_name, rule_name, severity, "
                                "extracted_value, expected_value, message, file_path) "
                                "VALUES (%s, %s, 'file_path', 'duplicate_document', 'warning', "
                                "%s, %s, %s, %s)",
                                (
                                    category, str(record_id), file_path,
                                    str(existing[0]),
                                    f"Duplicate upload: same file already extracted as record {existing[0]}",
                                    file_path,
                                ),
                            )
                            self._mark_extracted(record_id)
                            return
                finally:
                    conn.close()
            except Exception:
                logger.debug("Duplicate check failed", exc_info=True)

        try:
            nick = AgentNickOrchestrator(self._agent_nick)
            result = nick.process_document(
                file_path,
                category,
                user_id=str(user_id) if user_id is not None else None,
            )
            status = str(result.get("status", "error")).lower()
            if status in ("error", "failed"):
                raise RuntimeError(
                    f"Extraction {status}: {result.get('error', 'unknown')}"
                )
            self._mark_extracted(record_id)
            confidence = result.get("confidence", 0)
            error_count = result.get("errors", 0)
            pk = result.get("pk", "")
            doc_type = result.get("doc_type", "")

            logger.info(
                "Extraction completed for record %s: %s pk=%s fields=%s lines=%s discrep=%s conf=%s",
                record_id, doc_type, pk,
                result.get("header_fields"),
                result.get("line_items"),
                result.get("discrepancies"),
                confidence,
            )

            # --- PRE-KG VALIDATION GATE ---
            # Only sync to Knowledge Graph when extraction meets quality bar.
            # Bad data must not propagate to the graph where it would affect
            # downstream agents (ranking, opportunities, negotiation).
            kg_eligible = True
            if not pk:
                logger.warning(
                    "KG BLOCKED for record %s: missing primary key — data not synced to graph",
                    record_id,
                )
                kg_eligible = False
            elif confidence < 0.70:
                logger.warning(
                    "KG BLOCKED for record %s: confidence %.2f below 0.70 threshold — data not synced to graph",
                    record_id, confidence,
                )
                kg_eligible = False
            elif error_count > 2:
                logger.warning(
                    "KG BLOCKED for record %s: %d errors — data not synced to graph",
                    record_id, error_count,
                )
                kg_eligible = False

            if kg_eligible:
                try:
                    from services.procurement_kg_builder import ProcurementKGBuilder
                    builder = ProcurementKGBuilder(self._agent_nick)
                    builder.build_full_graph()
                    builder.close()
                    logger.info("KG synced after extraction of record %s", record_id)
                except Exception:
                    logger.debug("KG sync after extraction failed", exc_info=True)

            # --- TRAINING DATA COLLECTION ---
            # High-confidence, error-free extractions are automatically
            # collected as verified training examples for recursive fine-tuning.
            if confidence >= 0.90 and error_count == 0 and pk:
                try:
                    self._collect_training_example(
                        record_id, doc_type, pk, result, file_path,
                    )
                except Exception:
                    logger.warning("Training data collection failed", exc_info=True)

        except Exception as exc:
            logger.exception("Extraction failed for record %s", record_id)
            self._mark_failed(record_id, str(exc))

    def _collect_training_example(
        self, record_id: int, doc_type: str, pk: str,
        result: dict, file_path: str,
    ) -> None:
        """Collect verified extraction as training data for recursive fine-tuning.

        High-confidence, error-free extractions are appended to the
        fine-tuning dataset so the model learns from its own best work.
        """
        import json as _json
        training_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "training", "auto_collected_examples.jsonl",
        )
        os.makedirs(os.path.dirname(training_path), exist_ok=True)

        # Deduplicate: skip if this doc_type+pk already in training data
        if os.path.exists(training_path):
            dedup_key = f'"{doc_type}"' + '.*' + f'"{pk}"'
            with open(training_path) as f:
                for line in f:
                    if f'"doc_type": "{doc_type}"' in line and f'"pk": "{pk}"' in line:
                        logger.debug(
                            "Training example already exists for %s pk=%s — skipping",
                            doc_type, pk,
                        )
                        return

        # Build the training conversation (instruction → response)
        source_text = result.get("_source_text", "") if isinstance(result, dict) else ""

        # Fetch persisted data from DB as the "verified" output
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    pk_map = {"Invoice": ("proc.bp_invoice", "invoice_id"),
                              "Purchase_Order": ("proc.bp_purchase_order", "po_id"),
                              "Quote": ("proc.bp_quote", "quote_id")}
                    table_info = pk_map.get(doc_type)
                    if not table_info:
                        return
                    table, pk_col = table_info
                    cur.execute(f"SELECT row_to_json(t) FROM {table} t WHERE {pk_col} = %s", (pk,))
                    row = cur.fetchone()
                    if not row:
                        return
                    header_data = row[0]

                    # Get line items
                    line_table_map = {"Invoice": ("proc.bp_invoice_line_items", "invoice_id"),
                                     "Purchase_Order": ("proc.bp_po_line_items", "po_id"),
                                     "Quote": ("proc.bp_quote_line_items", "quote_id")}
                    lt_info = line_table_map.get(doc_type)
                    line_data = []
                    if lt_info:
                        lt_table, lt_fk = lt_info
                        cur.execute(f"SELECT row_to_json(t) FROM {lt_table} t WHERE {lt_fk} = %s", (pk,))
                        line_data = [r[0] for r in cur.fetchall()]
            finally:
                conn.close()
        except Exception:
            logger.debug("Failed to fetch persisted data for training", exc_info=True)
            return

        # Remove audit columns from training data
        skip = {"created_date", "created_by", "last_modified_date", "last_modified_by",
                "confidence_score", "needs_review", "ai_flag_required"}
        header_clean = {k: v for k, v in header_data.items() if k not in skip and v is not None}
        lines_clean = [{k: v for k, v in li.items() if k not in skip and v is not None} for li in line_data]

        example = {
            "doc_type": doc_type,
            "pk": pk,
            "file_path": file_path,
            "confidence": result.get("confidence", 0),
            "source_text": source_text[:6000] if source_text else "",
            "extracted": {"header": header_clean, "line_items": lines_clean},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(training_path, "a") as f:
            f.write(_json.dumps(example, default=str) + "\n")

        logger.info(
            "Collected training example: %s pk=%s conf=%.2f → %s",
            doc_type, pk, result.get("confidence", 0), training_path,
        )

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
        """Query for Completed/Running records and stale Extracting records, then dispatch."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Recover stale Extracting records (stuck > 30 minutes)
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
                    WHERE status IN ('Completed', 'Running')
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
