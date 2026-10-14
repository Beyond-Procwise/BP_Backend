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


class _RecordingCursor:
    """Cursor double that records executed SQL and serves scripted fetchone rows."""
    def __init__(self, fetch_script):
        # fetch_script: list of rows returned by successive fetchone() calls
        self._script = list(fetch_script)
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._script.pop(0) if self._script else None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _RecordingConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


class TestDocAction:
    def test_content_duplicate_marks_doc_action(self, dummy_nick):
        w = ProcessMonitorWatcher(dummy_nick)
        # fetchone script: 1) prior row with same hash (id, file_path, category)
        cur = _RecordingCursor(fetch_script=[(7, "documents/po/orig.pdf", "po")])
        conn = _RecordingConn(cur)
        with patch.object(w, "_get_connection", return_value=conn), \
             patch("src.services.extraction.content_hash.compute_content_hash",
                   return_value="abc123"), \
             patch.object(w, "_data_needs_reextraction", return_value=False) as gate, \
             patch.object(w, "_await_file", return_value=True), \
             patch.object(w, "_mark_extracted") as mark_ext:
            with w._processing_lock:
                w._processing_ids.add(42)
            w._process_record({"id": 42, "file_path": "documents/po/dup.pdf",
                               "category": "po", "user_id": 1,
                               "session_id": "S-1"})
        # The gate traces the EARLIER upload (id 7), not a file name.
        gate.assert_called_once_with(cur, 7, "po")
        sqls = " || ".join(s for s, _ in cur.executed)
        assert "doc_action = 'duplicate'" in sqls or "doc_action='duplicate'" in sqls
        # Session outcome recorded against THIS record's session_id (not by
        # file_path, which is ambiguous for a re-upload), then resolved.
        assert "session_document_outcome" in sqls
        assert "fn_try_resolve_session" in sqls
        assert any(p and "S-1" in p for _, p in cur.executed if p)
        mark_ext.assert_called_once_with(42)

    def test_content_duplicate_without_session_skips_outcome(self, dummy_nick):
        w = ProcessMonitorWatcher(dummy_nick)
        cur = _RecordingCursor(fetch_script=[(7, "documents/po/orig.pdf", "po")])
        conn = _RecordingConn(cur)
        with patch.object(w, "_get_connection", return_value=conn), \
             patch("src.services.extraction.content_hash.compute_content_hash",
                   return_value="abc123"), \
             patch.object(w, "_data_needs_reextraction", return_value=False), \
             patch.object(w, "_await_file", return_value=True), \
             patch.object(w, "_mark_extracted") as mark_ext:
            with w._processing_lock:
                w._processing_ids.add(42)
            w._process_record({"id": 42, "file_path": "documents/po/dup.pdf",
                               "category": "po", "user_id": 1})  # no session_id
        sqls = " || ".join(s for s, _ in cur.executed)
        assert "doc_action = 'duplicate'" in sqls or "doc_action='duplicate'" in sqls
        assert "session_document_outcome" not in sqls  # nothing to resolve
        mark_ext.assert_called_once_with(42)

    def test_no_prior_hash_proceeds_to_extract(self, dummy_nick):
        w = ProcessMonitorWatcher(dummy_nick)
        cur = _RecordingCursor(fetch_script=[None])  # no prior row
        conn = _RecordingConn(cur)
        # One pipeline now: src/services/extraction/. The EXTRACTION_RENOVATION_ENABLED
        # flag and the legacy extraction_v3 dispatch behind it are gone.
        with patch.object(w, "_get_connection", return_value=conn), \
             patch("src.services.extraction.content_hash.compute_content_hash",
                   return_value="abc123"), \
             patch.object(w, "_await_file", return_value=True), \
             patch("src.services.extraction.dispatch.dispatch_document",
                   return_value={"status": "promoted", "pk": "PO123",
                                 "confidence": 0.95, "errors": 0}) as disp, \
             patch.object(w, "_mark_extracted"), \
             patch.object(w, "_stamp_quality_action") as stamp:
            with w._processing_lock:
                w._processing_ids.add(43)
            w._process_record({"id": 43, "file_path": "documents/po/new.pdf",
                               "category": "po", "user_id": 1})
        disp.assert_called_once()
        stamp.assert_called_once()

    def test_stamp_quality_action_updated(self, dummy_nick):
        w = ProcessMonitorWatcher(dummy_nick)
        # prior row with same file_path, different hash → 'updated'
        cur = _RecordingCursor(fetch_script=[(9,)])
        conn = _RecordingConn(cur)
        with patch.object(w, "_get_connection", return_value=conn):
            w._stamp_quality_action(
                44, "documents/po/x.pdf", "hashNEW",
                {"confidence": 0.95, "pk": "PO9", "missing": []})
        sqls = " || ".join(s for s, _ in cur.executed)
        assert "doc_action" in sqls
        assert any(p and "updated" in p for _, p in cur.executed if p)

    def test_stamp_quality_action_needs_review(self, dummy_nick):
        w = ProcessMonitorWatcher(dummy_nick)
        cur = _RecordingCursor(fetch_script=[])
        conn = _RecordingConn(cur)
        with patch.object(w, "_get_connection", return_value=conn):
            w._stamp_quality_action(
                45, "documents/po/y.pdf", "h",
                {"confidence": 0.4, "pk": "PO1", "missing": []})
        assert any(p and "needs_review" in p for _, p in cur.executed if p)


class TestGraphSync:
    def test_an_extraction_does_not_rebuild_the_whole_graph(self, dummy_nick):
        """The graph mirrors _trgt and is synced per promoted row by the
        scheduler. A clean, high-confidence extraction has only reached _stg,
        so the watcher must not trigger a full Neo4j rebuild for it."""
        w = ProcessMonitorWatcher(dummy_nick)
        cur = _RecordingCursor(fetch_script=[None])
        conn = _RecordingConn(cur)
        with patch.object(w, "_get_connection", return_value=conn), \
             patch("src.services.extraction.content_hash.compute_content_hash",
                   return_value="abc123"), \
             patch.object(w, "_await_file", return_value=True), \
             patch("src.services.extraction.dispatch.dispatch_document",
                   return_value={"status": "promoted", "pk": "PO123",
                                 "confidence": 0.95, "errors": 0}), \
             patch.object(w, "_mark_extracted"), \
             patch.object(w, "_stamp_quality_action"), \
             patch.object(w, "_collect_training_example"), \
             patch("services.procurement_kg_builder.ProcurementKGBuilder") as builder:
            with w._processing_lock:
                w._processing_ids.add(46)
            w._process_record({"id": 46, "file_path": "documents/po/new.pdf",
                               "category": "po", "user_id": 1})
        builder.assert_not_called()


class TestReextractionGate:
    """A re-upload identical to an earlier one is skipped as a duplicate unless
    the earlier copy's data is provably gone. "Provably": its _raw row was
    promoted, and that document is in neither _stg nor _trgt any more."""

    def _gate(self, fetch_script, category="invoice"):
        cur = _RecordingCursor(fetch_script=fetch_script)
        needs = ProcessMonitorWatcher._data_needs_reextraction(cur, 7, category)
        return needs, " || ".join(s for s, _ in cur.executed), cur

    def test_promoted_document_that_is_gone_is_extracted_again(self):
        # raw row promoted → not found in _stg → not found in _trgt
        needs, _, _ = self._gate([("INV-001",), None, None])
        assert needs is True

    def test_promoted_document_still_held_is_a_duplicate(self):
        needs, _, _ = self._gate([("INV-001",), (1,)])
        assert needs is False

    def test_traces_the_earlier_upload_not_the_file_name(self):
        _, sql, cur = self._gate([("INV-001",), None, None])
        assert "proc.bp_invoice_raw" in sql
        assert "process_monitor_id" in sql
        assert cur.executed[0][1][0] == 7
        assert "proc.bp_invoice_stg" in sql and "proc.bp_invoice_trgt" in sql

    def test_never_queries_tables_that_do_not_exist(self):
        for cat in ("invoice", "po", "quote", "contract"):
            _, sql, _ = self._gate([("X-1",), None, None], category=cat)
            for bare in ("proc.bp_invoice ", "proc.bp_quote ", "proc.bp_purchase_order "):
                assert bare not in sql + " "

    def test_contract_is_checked_where_contracts_promote(self):
        _, sql, _ = self._gate([("C-1",), None], category="contract")
        assert "proc.bp_contract_raw" in sql and "proc.bp_contracts" in sql

    def test_unpromoted_or_untraceable_earlier_copy_stays_a_duplicate(self):
        # No promoted raw row: held for review, no PK, or nothing traceable.
        needs, _, _ = self._gate([None])
        assert needs is False

    def test_unknown_category_stays_a_duplicate(self):
        needs, sql, _ = self._gate([], category="brochure")
        assert needs is False and sql == ""
