"""Regression tests for RAGPipeline._is_internal_payload.

Uploaded chat attachments were embedded successfully but never retrievable.
RAGService.search() scopes the uploaded-documents collection with
FieldCondition(key="session_id"), so every retrievable attachment carries a
`session_id` payload field -- but _is_internal_payload() rejected any payload
holding a key containing "session_", so each hit was discarded as internal.
The two behaviours were mutually exclusive and the feature could never work.
"""

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.model_selector import RAGPipeline


def _make_pipeline(learning_collection: str = "learning") -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.rag = SimpleNamespace(
        learning_collection=learning_collection,
        primary_collection="procwise_document_embeddings",
        uploaded_collection="uploaded_documents",
    )
    return pipeline


def test_session_id_payload_is_not_internal():
    """An uploaded attachment carries session_id and must stay retrievable."""
    pipeline = _make_pipeline()
    payload = {
        "document_id": "cee82c45-fa13-46e0-aae6-4d14ae25e208",
        "session_id": "ses-20260710-E2SG",
        "doc_name": "invoice.pdf",
        "content": "TOTAL: £318.00",
    }
    assert pipeline._is_internal_payload(payload, "uploaded_documents") is False


def test_other_session_prefixed_keys_remain_internal():
    """Only the exact `session_id` key is exempt; internals stay blocked."""
    pipeline = _make_pipeline()
    for internal_key in ("session_state", "session_trace", "session_events"):
        payload = {"document_id": "x", internal_key: "whatever"}
        assert pipeline._is_internal_payload(payload, "uploaded_documents") is True


def test_workflow_and_agent_markers_still_internal():
    pipeline = _make_pipeline()
    for internal_key in ("workflow_id", "agent_name", "trace_id", "dispatch_at"):
        payload = {"document_id": "x", internal_key: "whatever"}
        assert pipeline._is_internal_payload(payload, "uploaded_documents") is True


def test_learning_collection_is_always_internal():
    pipeline = _make_pipeline()
    payload = {"document_id": "x", "session_id": "ses-1"}
    assert pipeline._is_internal_payload(payload, "learning") is True
