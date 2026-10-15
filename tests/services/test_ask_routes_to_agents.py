"""The ask pipeline hands a question to the agent the screen or the person names.

Built without its constructor, as test_ask_routes_analytic_questions does: the
branch under test runs before retrieval, so a pipeline with a history manager,
an agent registry and nothing else is enough. Reaching the general path is
proved the same way that file proves it: the stub has no retriever, so going
on to retrieval raises AttributeError.
"""

import types

import pytest

from src.services import ask_routing
from src.services.analytics import ask as analytics_ask
from src.services.tool_runtime import ToolRunResult


class _History:
    def __init__(self, items=None):
        self.items, self.saved = list(items or []), None

    def get_history(self, user_id):
        return list(self.items)

    def save_history(self, user_id, history):
        self.saved = history


class _Agent:
    def __init__(self, answer="Counter at £455.", raises=None):
        self.answer, self.raises, self.calls = answer, raises, []

    def reason(self, task, **kwargs):
        self.calls.append((task, kwargs))
        if self.raises:
            raise self.raises
        return ToolRunResult(answer=self.answer)


def _pipeline(agents, history=None):
    from src.services.model_selector import RAGPipeline
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_llm_model = "test-model"
    p.history_manager = _History(history)
    p.agent_nick = types.SimpleNamespace(agents=agents)
    return p


@pytest.fixture(autouse=True)
def _no_analytic_no_audit(monkeypatch):
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: None)
    monkeypatch.setattr(ask_routing, "screen_agent_map",
                        lambda read=None: {"negotiations": "negotiation"})
    monkeypatch.setattr("src.services.model_selector.record_ask_route", lambda **k: None)


def test_the_screen_agent_answers():
    agent = _Agent()
    result = _pipeline({"negotiation": agent}).answer_question(
        query="what should I counter at?", user_id="nick", screen="negotiations")
    assert result["answer"] == "Counter at £455."
    assert result["answered_by"] == {"agent": "negotiation", "label": "Negotiation agent",
                                     "source": "screen", "note": None}
    assert agent.calls[0][1]["user_id"] == "nick"


def test_the_chosen_agent_beats_the_screen():
    neg, quotes = _Agent("neg"), _Agent("quotes")
    result = _pipeline({"negotiation": neg, "quote_comparison": quotes}).answer_question(
        query="q", user_id="u", screen="negotiations", agent="quote_comparison")
    assert result["answer"] == "quotes" and not neg.calls


def test_an_analytic_question_beats_the_screen_agent(monkeypatch):
    payload = {"answer": "<section>top 10</section>", "follow_ups": [],
               "retrieved_documents": [], "next_steps": []}
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: dict(payload))
    agent = _Agent()
    result = _pipeline({"negotiation": agent}).answer_question(
        query="top 10 suppliers by spend", user_id="u", screen="negotiations")
    assert result["answer"] == payload["answer"]
    assert result["answered_by"]["source"] == "records"
    assert not agent.calls


def test_a_failing_agent_goes_on_to_the_general_path():
    pipeline = _pipeline({"negotiation": _Agent(raises=RuntimeError("down"))})
    with pytest.raises(AttributeError):
        pipeline.answer_question(query="q", user_id="u", screen="negotiations")


def test_a_slug_not_loaded_answers_generally():
    with pytest.raises(AttributeError):
        _pipeline({}).answer_question(query="q", user_id="u", agent="deleted_agent")


def test_a_question_with_an_uploaded_file_is_answered_generally():
    agent = _Agent()
    with pytest.raises(AttributeError):
        _pipeline({"negotiation": agent}).answer_question(
            query="q", user_id="u", screen="negotiations", files=[(b"x", "a.pdf")])
    assert not agent.calls


def test_the_answer_is_kept_in_the_history():
    pipeline = _pipeline({"negotiation": _Agent()})
    pipeline.answer_question(query="q", user_id="u", screen="negotiations")
    assert pipeline.history_manager.saved[-1]["answer"] == "Counter at £455."


def test_recent_turns_reach_the_agent_as_context():
    agent = _Agent()
    _pipeline({"negotiation": agent},
              history=[{"query": "who is the supplier?", "answer": "Acme Ltd"}]).answer_question(
        query="and their last offer?", user_id="u", screen="negotiations")
    assert "Acme Ltd" in agent.calls[0][1]["extra_system"]


def test_the_route_is_announced_before_the_work():
    events = []
    _pipeline({"negotiation": _Agent()}).answer_question(
        query="q", user_id="u", screen="negotiations",
        on_event=lambda kind, payload: events.append((kind, payload)))
    stages = [p.get("stage") for k, p in events if k == "stage"]
    assert stages[:2] == ["routed", "agent_working"]
    assert events[0][1]["answered_by"]["agent"] == "negotiation"
    assert ("delta", {"text": "Counter at £455."}) in events


def test_no_agent_and_no_screen_is_general_as_before():
    with pytest.raises(AttributeError):
        _pipeline({"negotiation": _Agent()}).answer_question(query="q", user_id="u")


def test_a_fallback_note_is_stamped_on_the_general_answer(monkeypatch):
    from src.services import model_selector

    real = model_selector.RAGPipeline._answer_general

    def _general_after_routing(self, **kw):
        # Run the real routing branch; stand in only for retrieval, which this stub cannot do.
        try:
            return real(self, **kw)
        except AttributeError:
            return {"answer": "general"}

    monkeypatch.setattr(model_selector.RAGPipeline, "_answer_general", _general_after_routing)
    pipeline = _pipeline({"negotiation": _Agent(raises=RuntimeError("down"))})
    result = pipeline.answer_question(query="q", user_id="u", screen="negotiations")
    assert result["answer"] == "general"
    assert result["answered_by"]["source"] == "fallback"
    assert result["answered_by"]["note"] == (
        "The Negotiation agent could not answer, so this was answered generally.")
