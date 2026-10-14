"""The ask path hands an analytic question to the analytic layer.

Everything under services/analytics was built without a caller. This is the
test that it has one: the /ask pipeline asks the seam first, returns its answer
whole when it gets one, and carries on down the retrieval path when it does not.

The pipeline is built without its constructor on purpose. answer_question's
first act is to consult the seam, so a test needs the two attributes read
before that and nothing else — a full RAGPipeline would drag Qdrant, Ollama and
the database into a test about a branch.
"""

import pytest

from src.services.analytics import ask as analytics_ask


class _History:
    def __init__(self):
        self.saved = None

    def get_history(self, user_id):
        return []

    def save_history(self, user_id, history):
        self.saved = history


def _pipeline():
    from src.services.model_selector import RAGPipeline

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.default_llm_model = "test-model"
    pipeline.history_manager = _History()
    return pipeline


PAYLOAD = {"answer": "<section class=\"agent-answer\">…</section>", "follow_ups": [],
           "retrieved_documents": [], "next_steps": [{"action_id": "analytic.x"}],
           "analytic_answer": {"answer_id": "a-1"}}


def test_an_analytic_answer_is_returned_whole(monkeypatch):
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: PAYLOAD)
    result = _pipeline().answer_question(query="top 10 suppliers by spend", user_id="u1")
    assert result == PAYLOAD


def test_the_question_the_currency_and_the_persona_reach_the_layer(monkeypatch):
    seen = {}

    def _capture(query, **kwargs):
        seen["query"] = query
        seen.update(kwargs)
        return PAYLOAD

    monkeypatch.setattr(analytics_ask, "analytic_answer", _capture)
    _pipeline().answer_question(query="top 10 suppliers by spend", user_id="u1",
                                display_currency="USD", persona="cpo",
                                action_id="analytic.supplier_concentration")
    assert seen["query"] == "top 10 suppliers by spend"
    assert seen["persona"] == "cpo"
    assert seen["display_currency"] == "USD"
    assert seen["action_id"] == "analytic.supplier_concentration"


def test_the_answer_is_kept_so_the_next_turn_can_refer_to_it(monkeypatch):
    # "Put that into a table" is answered from the previous turn. An analytic
    # answer that never entered the history would leave that question with
    # nothing to re-present.
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: PAYLOAD)
    pipeline = _pipeline()
    pipeline.answer_question(query="top 10 suppliers by spend", user_id="u1")
    assert pipeline.history_manager.saved[-1]["answer"] == PAYLOAD["answer"]


def test_a_question_the_layer_declines_goes_down_the_old_path(monkeypatch):
    # The seam returning None must not short-circuit anything. This stub
    # pipeline has no retriever, so reaching for one is the proof it continued.
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: None)
    with pytest.raises(AttributeError):
        _pipeline().answer_question(query="what is our payment terms policy?", user_id="u1")
