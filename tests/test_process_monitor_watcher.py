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
    def test_process_record_calls_agent_nick(self, dummy_nick, mock_orchestrator):
        """AgentNick is dispatched as primary agent for document processing."""
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        watcher._processing_ids.add(1)

        mock_result = {"status": "success", "doc_type": "Purchase_Order", "pk": "PO-001",
                       "header_fields": 10, "line_items": 3, "missing_fields": []}

        with patch("services.process_monitor_watcher.AgentNickOrchestrator") as MockNick:
            MockNick.return_value.process_document.return_value = mock_result
            with patch.object(watcher, "_mark_extracted") as mark_ok:
                watcher._process_record({
                    "id": 1,
                    "file_path": "documents/po/test.pdf",
                    "category": "po",
                })
                MockNick.return_value.process_document.assert_called_once_with(
                    "documents/po/test.pdf", "po", user_id=None,
                )
                mark_ok.assert_called_once_with(1)

    def test_process_record_marks_failed_on_error(self, dummy_nick, mock_orchestrator):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=mock_orchestrator)
        watcher._processing_ids.add(1)

        with patch("services.process_monitor_watcher.AgentNickOrchestrator") as MockNick:
            MockNick.return_value.process_document.return_value = {
                "status": "error", "error": "boom"
            }
            with patch.object(watcher, "_mark_failed") as mark_fail:
                watcher._process_record({
                    "id": 1,
                    "file_path": "documents/po/test.pdf",
                    "category": "po",
                })
                mark_fail.assert_called_once()
                assert "boom" in mark_fail.call_args[0][1]

    def test_process_record_handles_agent_nick_exception(self, dummy_nick):
        watcher = ProcessMonitorWatcher(dummy_nick, orchestrator=None)
        watcher._processing_ids.add(1)

        with patch("services.process_monitor_watcher.AgentNickOrchestrator") as MockNick:
            MockNick.return_value.process_document.side_effect = RuntimeError("crash")
            with patch.object(watcher, "_mark_failed") as mark_fail:
                watcher._process_record({
                    "id": 1,
                    "file_path": "documents/po/test.pdf",
                    "category": "po",
                })
                mark_fail.assert_called_once()
                assert "crash" in mark_fail.call_args[0][1]


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
