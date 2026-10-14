"""The watcher's gate, driven end to end with a fake dispatch.

The routing tests prove pipeline_for_category; they cannot prove the watcher
uses it, nor that a refusal is still recorded the way operators and the UI
expect (doc_action='unsupported'). That status was once derived from the old
message's wording, and a rewording silently dropped it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.process_monitor_watcher import ProcessMonitorWatcher  # noqa: E402


class _Cur:
    description = None

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _Conn:
    def cursor(self):
        return _Cur()

    def close(self):
        pass


def _run(category):
    w = ProcessMonitorWatcher(SimpleNamespace(settings=SimpleNamespace()))
    with patch.object(w, "_get_connection", return_value=_Conn()), \
         patch("src.services.extraction.content_hash.compute_content_hash",
               return_value="h"), \
         patch.object(w, "_await_file", return_value=True), \
         patch("src.services.extraction.dispatch.dispatch_document",
               return_value={"status": "promoted", "pk": "X1",
                             "confidence": 0.5, "errors": 0}) as disp, \
         patch.object(w, "_mark_extracted"), \
         patch.object(w, "_stamp_quality_action"), \
         patch.object(w, "_mark_failed") as failed:
        w._process_record({"id": 7, "file_path": "documents/x.pdf",
                           "category": category, "user_id": 1})
    return disp, failed


@pytest.mark.parametrize("category", ["bill of lading", "", None, "general notice"])
def test_a_refused_category_is_marked_failed_as_unsupported(category):
    disp, failed = _run(category)
    disp.assert_not_called()
    failed.assert_called_once()
    assert failed.call_args.kwargs.get("doc_action") == "unsupported"


@pytest.mark.parametrize("category,doc_type", [
    ("PO", "purchase_order"), ("Invoice", "invoice"),
    ("quotes", "quote"), ("framework agreement", "contract"),
])
def test_a_good_category_reaches_dispatch_with_the_right_doc_type(category, doc_type):
    disp, failed = _run(category)
    failed.assert_not_called()
    disp.assert_called_once()
    assert disp.call_args.kwargs["doc_type"] == doc_type
