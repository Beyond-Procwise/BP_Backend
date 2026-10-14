"""The watcher hands dispatch the concept the uploader declared, alongside the
pipeline it routed to. Routing itself is unchanged."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.routing import pipeline_for_category  # noqa: E402
from src.services.process_monitor_watcher import ProcessMonitorWatcher  # noqa: E402
from tests.services.concepts.test_gate_wiring import _Conn  # noqa: E402


@pytest.mark.parametrize("category", ["PO", "Invoice", "quotes", "framework agreement"])
def test_dispatch_receives_the_declared_concept_the_router_returned(category):
    doc_type, concept = pipeline_for_category(category)
    assert concept, "fixture must declare a concept or this proves nothing"
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
         patch.object(w, "_mark_failed"):
        w._process_record({"id": 7, "file_path": "documents/x.pdf",
                           "category": category, "user_id": 1})
    kw = disp.call_args.kwargs
    assert kw["doc_type"] == doc_type
    assert kw["declared_concept"] == concept
