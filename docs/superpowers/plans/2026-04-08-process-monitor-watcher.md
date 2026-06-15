# Process Monitor Watcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continuously monitor `proc.process_monitor` for completed document uploads and automatically trigger data extraction in real-time using PostgreSQL LISTEN/NOTIFY with concurrent processing.

**Architecture:** A `ProcessMonitorWatcher` service with a PG LISTEN thread for real-time notifications, a 60s fallback poller for resilience, and a `ThreadPoolExecutor` (4 workers) for concurrent extraction. Integrates into the existing `BackendScheduler` lifecycle. An atomic claim pattern (`UPDATE ... WHERE status='Completed' RETURNING *`) prevents duplicate processing.

**Tech Stack:** psycopg2 (existing), `concurrent.futures.ThreadPoolExecutor` (stdlib), `select` (stdlib for LISTEN), PostgreSQL triggers.

---

### Task 1: Create ProcessMonitorWatcher Service

**Files:**
- Create: `src/services/process_monitor_watcher.py`

- [ ] **Step 1: Create the service file with imports and class skeleton**

```python
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
        """Create the PG trigger + function if they don't exist (idempotent)."""
        sql = """
        CREATE OR REPLACE FUNCTION proc.notify_process_monitor_ready()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.status = 'Completed' THEN
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
                    cur.execute(sql)
                logger.info("PG trigger 'trg_process_monitor_ready' ensured")
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to ensure process_monitor trigger")

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
        logger.info(
            "Starting extraction for record %s: file_path=%s category=%s",
            record_id,
            file_path,
            category,
        )
        try:
            orchestrator = self._orchestrator
            if orchestrator is None:
                raise RuntimeError("Orchestrator not available")
            result = orchestrator.execute_extraction_flow(
                s3_object_key=file_path,
            )
            status = "error"
            if isinstance(result, dict):
                status = str(result.get("status", "error")).lower()
            if status in ("blocked",):
                raise RuntimeError(f"Extraction blocked: {result.get('reason', 'unknown')}")
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
        """Query for all Completed records and dispatch them."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
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
```

- [ ] **Step 2: Verify the file was created correctly**

Run: `python -c "import ast; ast.parse(open('src/services/process_monitor_watcher.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add src/services/process_monitor_watcher.py
git commit -m "feat: add ProcessMonitorWatcher service for real-time extraction dispatch"
```

---

### Task 2: Integrate into BackendScheduler

**Files:**
- Modify: `src/services/backend_scheduler.py:1-10` (imports)
- Modify: `src/services/backend_scheduler.py:50-71` (`__init__`)
- Modify: `src/services/backend_scheduler.py:148-161` (`stop`)

- [ ] **Step 1: Add import at top of backend_scheduler.py**

Add after the existing imports (line 6, after `from services.email_watcher import EmailWatcherService`):

```python
from services.process_monitor_watcher import ProcessMonitorWatcher
```

- [ ] **Step 2: Add watcher fields to `__init__`**

After line 62 (`self._email_watcher_lock = threading.Lock()`), add:

```python
        self._process_monitor_watcher: Optional[ProcessMonitorWatcher] = None
        self._process_monitor_lock = threading.Lock()
```

After line 71 (the `except` block for email watcher init), add:

```python
        try:
            self._ensure_process_monitor_watcher()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to start process monitor watcher during initialisation")
```

- [ ] **Step 3: Add `_ensure_process_monitor_watcher` and `get_process_monitor_watcher` methods**

Add after the `get_email_watcher_service` method (after line 278):

```python
    def _ensure_process_monitor_watcher(self) -> ProcessMonitorWatcher:
        """Create and start the ProcessMonitorWatcher if not already running."""
        with self._process_monitor_lock:
            if self._process_monitor_watcher is None:
                orchestrator = getattr(self, "_orchestrator", None)
                self._process_monitor_watcher = ProcessMonitorWatcher(
                    self.agent_nick,
                    orchestrator=orchestrator,
                )
                self._process_monitor_watcher.start()
            elif self._orchestrator is not None:
                self._process_monitor_watcher.update_orchestrator(self._orchestrator)
            return self._process_monitor_watcher

    def get_process_monitor_watcher(self) -> ProcessMonitorWatcher:
        """Expose the active ProcessMonitorWatcher instance."""
        return self._ensure_process_monitor_watcher()
```

- [ ] **Step 4: Add watcher shutdown to `stop` method**

In the `stop` method (after line 161, after the email watcher stop block), add:

```python
        with self._process_monitor_lock:
            monitor_watcher = self._process_monitor_watcher
        if monitor_watcher is not None:
            try:
                monitor_watcher.stop()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to stop process monitor watcher")
```

- [ ] **Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/services/backend_scheduler.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
git add src/services/backend_scheduler.py
git commit -m "feat: integrate ProcessMonitorWatcher into BackendScheduler lifecycle"
```

---

### Task 3: Wire into FastAPI Lifespan

**Files:**
- Modify: `src/api/main.py:47-58` (ProcwiseAppState Protocol)
- Modify: `src/api/main.py:116-138` (lifespan startup/shutdown)

- [ ] **Step 1: Add field to ProcwiseAppState Protocol**

After line 57 (`email_watcher_service: Optional[Any]`), add:

```python
    process_monitor_watcher: Optional[Any]
```

- [ ] **Step 2: Add watcher init in lifespan startup**

After line 124 (`state.email_watcher_owned = False`), add:

```python
        try:
            process_monitor_watcher = backend_scheduler.get_process_monitor_watcher()
        except Exception:
            logger.exception("Failed to obtain process monitor watcher from backend scheduler")
            process_monitor_watcher = None
        state.process_monitor_watcher = process_monitor_watcher
```

- [ ] **Step 3: Add to error-case defaults**

In the `except` block (around line 138), add after `state.backend_scheduler = None`:

```python
        state.process_monitor_watcher = None
```

- [ ] **Step 4: Add to lifespan teardown**

After the email watcher teardown block (after line 154), add:

```python
    if hasattr(state, "process_monitor_watcher"):
        state.process_monitor_watcher = None
```

- [ ] **Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/api/main.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py
git commit -m "feat: expose ProcessMonitorWatcher on FastAPI app state"
```

---

### Task 4: Write Tests

**Files:**
- Create: `tests/test_process_monitor_watcher.py`

- [ ] **Step 1: Write tests for the watcher service**

```python
"""Tests for ProcessMonitorWatcher service."""

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from services.process_monitor_watcher import ProcessMonitorWatcher


class DummyCursor:
    def __init__(self, rows=None, description=None):
        self._rows = rows or []
        self.description = description

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyConnection:
    def __init__(self, cursor=None):
        self._cursor = cursor or DummyCursor()
        self.autocommit = False
        self.notifies = []

    def cursor(self):
        return self._cursor

    def set_isolation_level(self, level):
        pass

    def poll(self):
        pass

    def close(self):
        pass

    def fileno(self):
        return 0


@pytest.fixture
def dummy_nick():
    return SimpleNamespace(
        settings=SimpleNamespace(
            db_host="localhost",
            db_name="testdb",
            db_user="user",
            db_password="pass",
            db_port=5432,
        )
    )


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.execute_extraction_flow.return_value = {"status": "success"}
    return orch


class TestClaimRecord:
    def test_claim_returns_record_on_success(self, dummy_nick, mock_orchestrator):
        row = (1, "Local Upload", "Upload", "Extracting", "documents/po/test.pdf",
               None, None, None, None, None, "po", "pdf", 30, 1)
        desc = [SimpleNamespace(name=n) for n in [
            "id", "process_name", "type", "status", "file_path",
            "start_ts", "created_date", "created_by",
            "lastmodified_date", "end_ts", "category",
            "document_type", "user_id", "total_count",
        ]]
        cursor = DummyCursor(rows=[row], description=desc)
        conn = DummyConnection(cursor=cursor)

        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        record = watcher._claim_record(conn, 1)
        assert record is not None
        assert record["id"] == 1
        assert record["file_path"] == "documents/po/test.pdf"

    def test_claim_returns_none_when_already_claimed(self, dummy_nick, mock_orchestrator):
        cursor = DummyCursor(rows=[], description=None)
        conn = DummyConnection(cursor=cursor)

        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        record = watcher._claim_record(conn, 1)
        assert record is None

    def test_claim_skips_already_processing_id(self, dummy_nick, mock_orchestrator):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        watcher._processing_ids.add(42)
        conn = DummyConnection()
        record = watcher._claim_record(conn, 42)
        assert record is None


class TestProcessRecord:
    def test_process_record_calls_extraction(self, dummy_nick, mock_orchestrator):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        watcher._processing_ids.add(1)

        with patch.object(watcher, "_mark_extracted") as mark_ok:
            watcher._process_record({
                "id": 1,
                "file_path": "documents/po/test.pdf",
                "category": "po",
            })
            mock_orchestrator.execute_extraction_flow.assert_called_once_with(
                s3_object_key="documents/po/test.pdf",
            )
            mark_ok.assert_called_once_with(1)

    def test_process_record_marks_failed_on_error(self, dummy_nick, mock_orchestrator):
        mock_orchestrator.execute_extraction_flow.side_effect = RuntimeError("boom")
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        watcher._processing_ids.add(1)

        with patch.object(watcher, "_mark_failed") as mark_fail:
            watcher._process_record({
                "id": 1,
                "file_path": "documents/po/test.pdf",
                "category": "po",
            })
            mark_fail.assert_called_once_with(1, "boom")

    def test_process_record_fails_without_orchestrator(self, dummy_nick):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=None)
        watcher._processing_ids.add(1)

        with patch.object(watcher, "_mark_failed") as mark_fail:
            watcher._process_record({
                "id": 1,
                "file_path": "documents/po/test.pdf",
                "category": "po",
            })
            mark_fail.assert_called_once()
            assert "Orchestrator not available" in mark_fail.call_args[0][1]


class TestLifecycle:
    def test_start_and_stop(self, dummy_nick, mock_orchestrator):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)

        with patch.object(watcher, "_ensure_trigger"):
            with patch.object(watcher, "_sweep_completed"):
                watcher.start()
                assert watcher._listen_thread is not None
                assert watcher._poll_thread is not None
                assert watcher._listen_thread.is_alive()
                assert watcher._poll_thread.is_alive()

                watcher.stop()
                assert not watcher._listen_thread.is_alive()
                assert not watcher._poll_thread.is_alive()

    def test_update_orchestrator(self, dummy_nick, mock_orchestrator):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=None)
        assert watcher._orchestrator is None
        watcher.update_orchestrator(mock_orchestrator)
        assert watcher._orchestrator is mock_orchestrator
```

- [ ] **Step 2: Run the tests**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/test_process_monitor_watcher.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_process_monitor_watcher.py
git commit -m "test: add ProcessMonitorWatcher unit tests"
```

---

### Task 5: End-to-End Verification

- [ ] **Step 1: Syntax-check all modified files**

Run:
```bash
python -c "
import ast
for f in ['src/services/process_monitor_watcher.py', 'src/services/backend_scheduler.py', 'src/api/main.py']:
    ast.parse(open(f).read())
    print(f'{f}: OK')
"
```
Expected: All three print OK

- [ ] **Step 2: Run full test suite**

Run: `cd /home/muthu/PycharmProjects/BP_Backend && python -m pytest tests/ -v --timeout=60`
Expected: All existing tests continue to pass, new tests pass

- [ ] **Step 3: Verify PG trigger creation against live database**

Run:
```bash
PGPASSWORD='Pr0cw!5edb001' psql -h "procwisemvpdb01.cluster-cpae0sg4mrk8.eu-west-1.rds.amazonaws.com" \
  -U "procwisedb123" -d "uicanvas" -p 5432 \
  -c "SELECT tgname FROM pg_trigger WHERE tgrelid = 'proc.process_monitor'::regclass"
```
Expected: Shows `trg_process_monitor_ready` (will be created on first service start)
