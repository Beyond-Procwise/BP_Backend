"""The process-monitor watcher embeds each document the live path extracts.

The embed is the last step and the only optional one: a vector index that is
down must not fail an extraction whose rows are already in the database. It
logs at ERROR instead, and /health's vector_store count shows the gap.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.process_monitor_watcher import ProcessMonitorWatcher


class _Cur:
    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return None

    def close(self):
        pass


class _Conn:
    def __init__(self):
        self.closed = False

    def cursor(self):
        return _Cur()

    def close(self):
        self.closed = True


@pytest.fixture
def nick():
    return SimpleNamespace(settings=SimpleNamespace(
        db_host="h", db_name="d", db_user="u", db_password="p", db_port=5432))


def _run(w, result):
    """Drive one record through _process_record with dispatch stubbed."""
    with patch.object(w, "_get_connection", return_value=_Conn()), \
         patch("src.services.extraction.content_hash.compute_content_hash", return_value=None), \
         patch.object(w, "_await_file", return_value=True), \
         patch("src.services.extraction.dispatch.dispatch_document", return_value=result), \
         patch.object(w, "_stamp_quality_action"), \
         patch.object(w, "_mark_extracted") as mark_ok:
        with w._processing_lock:
            w._processing_ids.add(43)
        w._process_record({"id": 43, "file_path": "documents/invoice/a.pdf",
                           "category": "invoice", "user_id": 1})
    return mark_ok


def test_a_completed_extraction_is_embedded(nick):
    w = ProcessMonitorWatcher(nick)
    result = {"status": "promoted", "doc_pk": "INV-1", "doc_type": "invoice",
              "raw_id": 7, "raw_persisted": True}

    with patch.object(w, "_embed_document") as emb:
        mark_ok = _run(w, result)

    emb.assert_called_once_with(result)
    mark_ok.assert_called_once_with(43)


def test_embed_document_passes_the_documents_type_key_and_raw_row(nick):
    w = ProcessMonitorWatcher(nick)
    conn = _Conn()
    with patch.object(w, "_get_connection", return_value=conn), \
         patch("src.services.extraction.embed.embed_document", return_value=3) as emb:
        w._embed_document({"doc_type": "invoice", "doc_pk": "INV-1", "raw_id": 7})

    _, kwargs = emb.call_args
    assert emb.call_args.args[0] is nick
    assert kwargs == {"doc_type": "invoice", "doc_pk": "INV-1", "raw_id": 7}
    assert conn.closed


def test_a_failed_embed_is_logged_and_never_fails_the_extraction(nick, caplog):
    w = ProcessMonitorWatcher(nick)
    with patch.object(w, "_get_connection", return_value=_Conn()), \
         patch("src.services.extraction.embed.embed_document",
               side_effect=RuntimeError("qdrant away")):
        w._embed_document({"doc_type": "invoice", "doc_pk": "INV-1", "raw_id": 7})

    assert any(r.levelname == "ERROR" and "embed" in r.getMessage().lower()
               for r in caplog.records)


@pytest.mark.parametrize("result", [
    {"doc_type": "invoice", "doc_pk": "", "raw_id": 7},
    {"doc_type": "invoice", "doc_pk": "INV-1", "raw_id": None},
    {"doc_type": "", "doc_pk": "INV-1", "raw_id": 7},
])
def test_nothing_to_embed_without_a_type_key_and_raw_row(nick, result):
    w = ProcessMonitorWatcher(nick)
    with patch.object(w, "_get_connection") as get_conn, \
         patch("src.services.extraction.embed.embed_document") as emb:
        w._embed_document(result)

    emb.assert_not_called()
    get_conn.assert_not_called()
