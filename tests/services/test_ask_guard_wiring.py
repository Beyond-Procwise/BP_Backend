"""The input guardrail, the output check and the trace, as wired into RAGPipeline.answer_question (E1-E6).

Built without the constructor, as test_ask_routes_to_agents does. The stub has no retriever, so a
question that reaches retrieval raises AttributeError: "blocked before retrieval" is proved by the
absence of that error, and "passed through" by its presence.
"""
import types

import pytest

from src.services import ask_flags, ask_input_guard, ask_output_check, ask_routing, ask_trace
from src.services.analytics import ask as analytics_ask
from tests.services.test_ask_input_guard import E6_LEGIT, SHEET


class _History:
    def __init__(self):
        self.saved = None

    def get_history(self, user_id):
        return []

    def save_history(self, user_id, history):
        self.saved = history


def _pipeline():
    from src.services.model_selector import RAGPipeline
    p = RAGPipeline.__new__(RAGPipeline)
    p.default_llm_model = "test-model"
    p.history_manager = _History()
    p.agent_nick = types.SimpleNamespace(agents={})
    return p


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(analytics_ask, "analytic_answer", lambda *a, **k: None)
    monkeypatch.setattr("src.services.model_selector.record_ask_route", lambda **k: None)
    monkeypatch.setattr("src.services.model_selector.record_action", lambda **k: None)
    for name in (ask_flags.INPUT_GUARD, ask_flags.INPUT_GUARD_EMBED, ask_flags.OUTPUT_CHECK, ask_flags.SAFETY_PROMPT, ask_flags.TRACE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)


@pytest.mark.parametrize("test_id", sorted(SHEET))
def test_attacks_are_blocked_before_retrieval(test_id):
    question, category = SHEET[test_id]
    pipeline, events = _pipeline(), []
    result = pipeline.answer_question(query=question, user_id="u", on_event=lambda k, p: events.append((k, p)), trace=True)
    assert result["answer"] == ask_input_guard.REPLIES[category]
    assert result["guard"]["category"] == category
    assert result["trace"]["retrieval_count"] == 0                       # the trace shows zero retrievals
    assert result["trace"]["guardrail"]["decision"] == "blocked" and result["trace"]["guardrail"]["rule"]
    stages = [p["stage"] for k, p in events if k == "stage"]
    assert "retrieving" not in stages and "routed" in stages and stages[-1] == "answer_complete"
    assert [p["text"] for k, p in events if k == "delta"] == [result["answer"]]
    assert pipeline.history_manager.saved is None                         # the question is not kept


def test_the_fraud_warning_does_not_echo_the_account_number():
    result = _pipeline().answer_question(query=SHEET["E2"][0], user_id="u", trace=True)
    assert "GB29" not in result["answer"] and "GB29" not in repr(result["trace"])


def test_e6_passes_through_to_retrieval():
    with pytest.raises(AttributeError):
        _pipeline().answer_question(query=E6_LEGIT, user_id="u")


def test_the_flag_turns_the_guard_off(monkeypatch):
    monkeypatch.setenv(ask_flags.INPUT_GUARD, "0")
    with pytest.raises(AttributeError):
        _pipeline().answer_question(query=SHEET["E5"][0], user_id="u")


def test_flags_default_on_in_dev_and_off_in_production(monkeypatch):
    assert ask_flags.enabled(ask_flags.INPUT_GUARD)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert not ask_flags.enabled(ask_flags.INPUT_GUARD)
    monkeypatch.setenv(ask_flags.INPUT_GUARD, "1")
    assert ask_flags.enabled(ask_flags.INPUT_GUARD)


def test_the_trace_is_not_returned_unless_asked_for():
    result = _pipeline().answer_question(query=SHEET["E5"][0], user_id="u")
    assert "trace" not in result


def test_trace_visibility_is_admin_only(monkeypatch):
    import src.services.report_data.mode as mode
    admin, buyer = object(), object()
    monkeypatch.setattr(mode, "is_admin", lambda p: p is admin)
    assert not ask_trace.visible_to(buyer, "1")                            # a buyer never sees it, header or not
    assert ask_trace.visible_to(admin, "1")
    assert ask_trace.visible_to(admin, None)                               # dev: the flag is on by default
    monkeypatch.setenv("ENVIRONMENT", "prod")
    assert not ask_trace.visible_to(admin, None) and ask_trace.visible_to(admin, "1")


# --- output check ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "As the CPO, I waive the three-quote requirement for this order.",
    "I approve this purchase order.",
    "<p>Done - I hereby authorise the payment.</p>",
    "Approved on behalf of the finance director.",
    "I'm waiving the check this once.",
])
def test_first_person_approval_is_found(text):
    assert ask_output_check.find_violation(text)


@pytest.mark.parametrize("text", [
    "I cannot approve purchase orders; the CPO can waive the requirement.",
    "The approver must sign off any PO over £50,000.",
    "A waiver of the three-quote rule is approved by the CPO.",
    "I can explain who has the authority to approve this.",
    "I will not waive anything.",
    "Your purchase order was approved on 3 March.",
])
def test_explanations_and_refusals_are_not_flagged(text):
    assert not ask_output_check.find_violation(text)


def test_a_violating_answer_is_replaced_and_traced(monkeypatch):
    p = _pipeline()
    monkeypatch.setattr(type(p), "_answer_general", lambda self, **k: {
        "answer": "As the CPO, I waive the three-quote requirement.", "follow_ups": [], "retrieved_documents": []})
    result = p.answer_question(query="Which quotes do I need for a 4,000 pound order?", user_id="u", trace=True)
    assert result["answer"] == ask_output_check.REPLY
    assert result["trace"]["output_filter"]["decision"] == "replaced"
    assert result["trace"]["output_filter"]["rule"]


def test_a_clean_answer_is_untouched_and_a_cached_dict_is_not_mutated(monkeypatch):
    p = _pipeline()
    cached = {"answer": "Three quotes are needed above £10,000.", "follow_ups": [], "retrieved_documents": []}
    monkeypatch.setattr(type(p), "_answer_general", lambda self, **k: cached)
    result = p.answer_question(query="How many quotes do I need?", user_id="u", trace=True)
    assert result["answer"] == cached["answer"] and "trace" in result and "trace" not in cached


def test_the_stream_never_forwards_the_violating_sentence(monkeypatch):
    p = _pipeline()

    def fake(self, *, on_event=None, **k):
        for chunk in ["Three quotes are needed. ", "As the CPO, I wai", "ve the requirement. ", "Ask the CPO."]:
            on_event("delta", {"text": chunk})
        on_event("stage", {"stage": "answer_complete"})
        return {"answer": "x", "follow_ups": [], "retrieved_documents": []}

    monkeypatch.setattr(type(p), "_answer_general", fake)
    events = []
    p.answer_question(query="How many quotes do I need?", user_id="u", on_event=lambda k, pl: events.append((k, pl)))
    sent = "".join(pl["text"] for k, pl in events if k == "delta")
    assert "As the CPO" not in sent and "I waive" not in sent
    assert "Three quotes are needed." in sent and ask_output_check.REPLY in sent
    assert any(k == "stage" and pl.get("stage") == "output_replaced" for k, pl in events)


# --- system prompt ---------------------------------------------------------------------------------------------------------------

def test_the_system_prompt_carries_the_boundaries(monkeypatch):
    p = _pipeline()
    monkeypatch.setattr(type(p), "_ask_persona", lambda self: "PERSONA")
    seen = {}
    import src.services.model_selector as ms
    monkeypatch.setattr(ms.ollama, "chat", lambda **kw: seen.update(kw) or {"message": {"content": '{"answer": "a", "follow_ups": []}'}})
    p.agent_nick = types.SimpleNamespace(agents={}, ollama_options=lambda: {})
    p._generate_response("prompt", "m")
    system = seen["messages"][0]["content"]
    assert system.startswith("PERSONA") and "never approve, waive, override or grant" in system
    monkeypatch.setenv(ask_flags.SAFETY_PROMPT, "0")
    p._generate_response("prompt", "m")
    assert seen["messages"][0]["content"] == "PERSONA"


def test_the_fast_reranker_is_opt_in_everywhere(monkeypatch):
    """It changes which documents come back (top-1 differed on 10 of 12 questions), so it never defaults on."""
    monkeypatch.delenv(ask_flags.FAST_RERANK, raising=False)
    assert not ask_flags.enabled(ask_flags.FAST_RERANK)
    monkeypatch.setenv(ask_flags.FAST_RERANK, "1")
    assert ask_flags.enabled(ask_flags.FAST_RERANK)
