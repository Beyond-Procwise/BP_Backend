"""The answer is finished before the reply is.

The model is asked for one JSON object holding both ``answer`` and
``follow_ups``. Only ``answer`` is streamed to the screen, so once its closing
quote arrives the user has read everything they are going to be shown — but the
model carries on writing the three follow-up questions, and the caller has no
way to know the prose is done until the whole object lands. Measured live that
tail is ~8.5s of a ~30s ask: a spinner turning over an answer that is already
complete.

These tests pin the notification that closes the gap: ``_generate_response``
tells the caller the moment the answer field closes, and ``answer_question``
forwards it to streaming clients as its own SSE stage.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services.model_selector import RAGPipeline


def _pipeline() -> RAGPipeline:
    """A pipeline shell — these tests drive _generate_response directly."""

    return RAGPipeline.__new__(RAGPipeline)


def _chat_stream(chunks: List[str]):
    """Stand in for ollama.chat(stream=True)."""

    def _chat(**kwargs):
        assert kwargs.get("stream") is True
        for chunk in chunks:
            yield {"message": {"content": chunk}}

    return _chat


# The answer closes at the end of chunk 2; everything after is follow-up tokens.
_CHUNKS = [
    '{"answer": "You have 123 suppliers',
    ' on record.", ',
    '"follow_ups": ["Which are active?", ',
    '"Shall I rank them?", "Any expiring?"]}',
]


def test_answer_complete_fires_once_when_the_answer_field_closes(monkeypatch):
    pipeline = _pipeline()
    monkeypatch.setattr(pipeline, "_ask_persona", lambda: "persona", raising=False)
    monkeypatch.setattr(
        pipeline, "agent_nick", SimpleNamespace(ollama_options=lambda: {}), raising=False
    )
    monkeypatch.setattr("services.model_selector.ollama.chat", _chat_stream(_CHUNKS))

    seen: List[str] = []
    completed: List[str] = []

    payload = pipeline._generate_response(
        "prompt",
        "model",
        on_delta=lambda text: seen.append(text),
        on_answer_complete=lambda text: completed.append(text),
    )

    assert completed == ["You have 123 suppliers on record."]
    assert "".join(seen) == "You have 123 suppliers on record."
    # The full object is still parsed and returned unchanged.
    assert payload["answer"] == "You have 123 suppliers on record."
    assert len(payload["follow_ups"]) == 3


def test_answer_complete_precedes_the_follow_up_tokens(monkeypatch):
    """Firing at the end is worthless — it must beat the follow-up generation."""

    pipeline = _pipeline()
    monkeypatch.setattr(pipeline, "_ask_persona", lambda: "persona", raising=False)
    monkeypatch.setattr(
        pipeline, "agent_nick", SimpleNamespace(ollama_options=lambda: {}), raising=False
    )

    timeline: List[str] = []

    def _chat(**kwargs):
        for chunk in _CHUNKS:
            timeline.append(f"chunk:{chunk[:12]}")
            yield {"message": {"content": chunk}}

    monkeypatch.setattr("services.model_selector.ollama.chat", _chat)

    pipeline._generate_response(
        "prompt",
        "model",
        on_delta=lambda text: None,
        on_answer_complete=lambda text: timeline.append("ANSWER_COMPLETE"),
    )

    marker = timeline.index("ANSWER_COMPLETE")
    remaining = timeline[marker + 1 :]
    # The follow-up chunks are still to come when the caller is told.
    assert any("follow_ups" in entry for entry in remaining), timeline


def test_missing_callback_is_not_required(monkeypatch):
    """Non-streaming callers pass neither hook and must be unaffected."""

    pipeline = _pipeline()
    monkeypatch.setattr(pipeline, "_ask_persona", lambda: "persona", raising=False)
    monkeypatch.setattr(
        pipeline, "agent_nick", SimpleNamespace(ollama_options=lambda: {}), raising=False
    )
    monkeypatch.setattr("services.model_selector.ollama.chat", _chat_stream(_CHUNKS))

    payload = pipeline._generate_response("prompt", "model", on_delta=lambda _t: None)

    assert payload["answer"] == "You have 123 suppliers on record."


def test_partial_answer_never_reports_complete(monkeypatch):
    """A truncated stream must not claim the answer finished.

    If the model dies mid-sentence the closing quote never arrives, and telling
    the client the prose is complete would freeze a half-written answer on
    screen as though it were the whole reply.
    """

    pipeline = _pipeline()
    monkeypatch.setattr(pipeline, "_ask_persona", lambda: "persona", raising=False)
    monkeypatch.setattr(
        pipeline, "agent_nick", SimpleNamespace(ollama_options=lambda: {}), raising=False
    )
    monkeypatch.setattr(
        "services.model_selector.ollama.chat",
        _chat_stream(['{"answer": "You have 123 supp']),
    )

    completed: List[str] = []
    pipeline._generate_response(
        "prompt",
        "model",
        on_delta=lambda _t: None,
        on_answer_complete=lambda text: completed.append(text),
    )

    assert completed == []
