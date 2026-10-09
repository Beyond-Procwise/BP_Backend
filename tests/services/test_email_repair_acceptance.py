"""The repair pass: which model it asks, and which repairs it keeps.

Found live 2026-10-09: the repair (and the counter compose) asked for 'mistral', which only worked because it is not
installed and call_ollama fell back to AgentNick; and a repair that removed two problems while ADDING a new placeholder
was accepted, because acceptance only counted failures.
"""

from types import SimpleNamespace

import pytest

from agents import email_drafting_agent as module
from agents.email_drafting_agent import EmailDraftingAgent

AGENTNICK = "BeyondProcwise/AgentNick:unified"


def test_the_drafting_agents_default_model_is_agentnick_not_another_model():
    assert module.DEFAULT_NEGOTIATION_MODEL == AGENTNICK


def test_the_repair_pass_asks_agentnick(monkeypatch):
    agent = EmailDraftingAgent()
    monkeypatch.delattr(agent.agent_nick.settings, "negotiation_email_model", raising=False)
    asked = {}

    def call_ollama(**kw):
        asked["model"] = kw["model"]
        return {"message": {"content": "Thank you for your offer. Please confirm the revised price by 30 October 2026."}}
    monkeypatch.setattr(agent, "call_ollama", call_ollama)
    agent._repair_assured_body("Body [name] text that is long enough to be a real email body.", [{"kind": "k", "detail": "d"}])
    assert asked["model"] == AGENTNICK


class Run:
    """A run whose checks are a fixed table: text -> failures."""

    def __init__(self, table):
        self.table, self.inputs, self.repair_rejected, self.repair_skipped = table, SimpleNamespace(reasoned={}), None, None

    def payment_hold(self, body):
        return None

    def check(self, body):
        return [{"kind": k, "detail": d, "severity": "fail"} for k, d in self.table.get(body, [])]


ORIGINAL = "Thank you for your offer of 47.50 GBP. We propose 42.00 GBP. Speak soon, [name]."
ADDS_ONE = "Thank you for your offer of 47.50 GBP. We propose 42.00 GBP. Please confirm by [deadline]."
CLEAN = "Thank you for your offer of 47.50 GBP. We propose 42.00 GBP. Please confirm by Friday? Kind regards."


def _agent_returning(monkeypatch, text):
    agent = EmailDraftingAgent()
    monkeypatch.setattr(agent, "_repair_assured_body", lambda body, failed, **k: text)
    return agent


def test_a_repair_that_adds_a_new_failure_is_rejected_even_if_the_count_goes_down(monkeypatch):
    run = Run({ORIGINAL: [("ungrounded_figure", "42.00"), ("unresolved_placeholder", "[name]"),
                          ("missing_required_element", "explicit_ask"), ("missing_required_element", "deadline")],
               ADDS_ONE: [("ungrounded_figure", "42.00"), ("unresolved_placeholder", "[deadline]"),
                          ("missing_required_element", "deadline")]})
    body, repaired = _agent_returning(monkeypatch, ADDS_ONE)._assure_composed(run, ORIGINAL)
    assert (body, repaired) == (ORIGINAL, False)
    assert "new problem" in run.repair_rejected and "[deadline]" in run.repair_rejected


def test_a_repair_that_only_removes_failures_is_kept(monkeypatch):
    run = Run({ORIGINAL: [("unresolved_placeholder", "[name]"), ("missing_required_element", "deadline")],
               CLEAN: []})
    body, repaired = _agent_returning(monkeypatch, CLEAN)._assure_composed(run, ORIGINAL)
    assert (body, repaired) == (CLEAN, True) and run.repair_rejected is None


# --- reading the model's reply ---------------------------------------------------------------------------------------------------
# The drafter's extractor required a dict, and the ollama client returns a ChatResponse object, so every real reply read as ""
# and every model-written email fell back to its template (found live 2026-10-09; the same bug was fixed in email_intent 2026-07-28).

def _chat_response(text):
    from ollama._types import ChatResponse, Message
    return ChatResponse(model="BeyondProcwise/AgentNick:unified", message=Message(role="assistant", content=text))


@pytest.mark.xfail(strict=True, reason="KNOWN BUG, held back 2026-10-09: the drafter's reader drops every ChatResponse. The one-line fix "
                   "(c389e9a9, reverted) waits until model-written drafts stop inventing names, addresses and phone numbers.")
def test_the_drafter_reads_a_real_ollama_chat_response():
    assert EmailDraftingAgent._extract_ollama_message(_chat_response("  Dear Sam, please confirm.  ")) == "Dear Sam, please confirm."


def test_the_drafter_still_reads_plain_dicts_in_both_shapes():
    assert EmailDraftingAgent._extract_ollama_message({"message": {"content": "chat"}}) == "chat"
    assert EmailDraftingAgent._extract_ollama_message({"response": "generate"}) == "generate"
    assert EmailDraftingAgent._extract_ollama_message(None) == ""


@pytest.mark.xfail(strict=True, reason="KNOWN BUG, held back 2026-10-09: the drafter's reader drops every ChatResponse. The one-line fix "
                   "(c389e9a9, reverted) waits until model-written drafts stop inventing names, addresses and phone numbers.")
def test_chat_returns_the_models_text_not_an_empty_string(monkeypatch):
    agent = EmailDraftingAgent()
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: _chat_response("Dear Sam, thank you for your quote."))
    assert module._chat(AGENTNICK, "system", "user", agent=agent) == "Dear Sam, thank you for your quote."


@pytest.mark.xfail(strict=True, reason="KNOWN BUG, held back 2026-10-09: the drafter's reader drops every ChatResponse. The one-line fix "
                   "(c389e9a9, reverted) waits until model-written drafts stop inventing names, addresses and phone numbers.")
def test_the_repair_pass_returns_the_models_repair(monkeypatch):
    agent = EmailDraftingAgent()
    fixed = "Thank you for your offer of 47.50 GBP. We propose 44.80 GBP. Please confirm by 30 October 2026."
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: _chat_response(fixed))
    out = agent._repair_assured_body("Thank you. [name]", [{"kind": "unresolved_placeholder", "detail": "[name]"}])
    assert out is not None and fixed in out                     # the body is wrapped in <p> by the sanitiser, as every draft is


def test_a_missing_deadline_we_do_not_hold_never_reaches_the_model(monkeypatch):
    agent = EmailDraftingAgent()
    calls = []
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: calls.append(kw) or _chat_response("x" * 50))
    assert agent._repair_assured_body("Thank you. We propose 44.80 GBP. Could you let us know?",
                                      [{"kind": "missing_required_element", "detail": "deadline", "severity": "fail"}]) is None
    assert calls == []


def test_the_model_is_told_the_deadline_we_hold_and_no_internal_codes(monkeypatch):
    agent = EmailDraftingAgent()
    calls = []
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: calls.append(kw) or _chat_response("x" * 50))
    agent._repair_assured_body("Body", [{"kind": "missing_required_element", "detail": "deadline", "severity": "fail"},
                                        {"kind": "ungrounded_figure", "detail": "43.10", "severity": "fail"}], deadline="30 October 2026")
    told = calls[0]["messages"][1]["content"]
    assert "30 October 2026" in told and "43.10" in told and "ungrounded_figure" not in told
