"""The two tool loops that predate the executor now ask the policy gate before every tool call and
reach the model through the driver. (Moving them onto the executor itself is a larger rewrite; until
then they may not run a tool the gate has not allowed.)"""
import json
import os
from types import SimpleNamespace

import pytest

from src.services import agentnick_driver as driver
from src.services import egress, tool_runtime
from src.services.governance_tools import governed_reasoning as GR
from src.services.supplier_enrichment import research as R

REFUSAL = {"result": "blocked", "reasonCode": "policy_blocked", "reason": "not allowed"}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- the helper ----------------------------------------------------------------------------------

def test_gate_refusal_is_none_when_the_gate_allows_and_the_refusal_when_it_does_not(monkeypatch):
    monkeypatch.setattr(tool_runtime, "_policy_refusal", lambda *a, **k: None)
    assert tool_runtime.gate_refusal("t", {}) is None
    monkeypatch.setattr(tool_runtime, "_policy_refusal", lambda *a, **k: REFUSAL)
    assert tool_runtime.gate_refusal("t", {}) == REFUSAL


def test_an_allow_on_precedent_is_an_allow_for_these_loops(monkeypatch):
    monkeypatch.setattr(tool_runtime, "_policy_refusal", lambda *a, **k: tool_runtime._RanOnPrecedent("note"))
    assert tool_runtime.gate_refusal("t", {}) is None


def test_gate_refusal_passes_who_and_why_to_the_gate(monkeypatch):
    seen = {}
    monkeypatch.setattr(tool_runtime, "_policy_refusal", lambda name, args, **k: seen.update(name=name, args=args, **k))
    tool_runtime.gate_refusal("get_policy", {"query": "x"}, agent="ag", reason="because", workflow_id="wf", user_id="u")
    assert seen == {"name": "get_policy", "args": {"query": "x"}, "agent": "ag", "reason": "because",
                    "workflow_id": "wf", "user_id": "u"}


# ---- governed reasoning ------------------------------------------------------------------------------

def _gr_setup(monkeypatch, gate, turns):
    monkeypatch.setattr(GR, "gate_refusal", gate)
    monkeypatch.setattr(GR.GT, "refresh", lambda: None)
    monkeypatch.setattr(GR.GT, "list_governance", lambda a=None: {"policies": [], "prompts": []})
    sent, ran = [], []
    it = iter(turns)

    def fake_chat(messages):
        sent.append(json.loads(json.dumps(messages, default=str)))
        return next(it)

    monkeypatch.setattr(GR, "_chat", fake_chat)
    monkeypatch.setattr(GR.GT, "get_policy", lambda q: (ran.append(q), {"policy_type": "t", "slug": "s"})[1])
    monkeypatch.setattr(GR, "_audit", lambda result, task, agent: result)
    return sent, ran


CALL = {"tool_calls": [{"function": {"name": "get_policy", "arguments": {"query": "supplier_ranking"}}}]}


def test_a_refused_governance_tool_is_never_run_and_the_model_is_told(monkeypatch):
    asked = []
    sent, ran = _gr_setup(monkeypatch, lambda name, args, **k: (asked.append((name, args, k.get("agent"))), REFUSAL)[1],
                          [CALL, {"content": "I could not fetch it."}])
    res = GR.govern("rank them", "ranker")
    assert ran == [] and asked == [("get_policy", {"query": "supplier_ranking"}, "ranker")]
    assert res["governance_used"]["policies"] == []
    tool_msg = [m for m in sent[1] if m.get("role") == "tool"][0]
    assert json.loads(tool_msg["content"]) == REFUSAL


def test_an_allowed_governance_tool_runs_as_before(monkeypatch):
    sent, ran = _gr_setup(monkeypatch, lambda *a, **k: None, [CALL, {"content": "Done."}])
    res = GR.govern("rank them", "ranker")
    assert ran == ["supplier_ranking"] and res["governance_used"]["policies"] == [{"policy_type": "t", "slug": "s"}]


def test_an_unknown_governance_tool_is_not_sent_to_the_gate(monkeypatch):
    asked = []
    _gr_setup(monkeypatch, lambda name, args, **k: asked.append(name), [
        {"tool_calls": [{"function": {"name": "mystery", "arguments": {}}}]}, {"content": "ok"}])
    GR.govern("t", None)
    assert asked == []


def test_governed_reasoning_reaches_the_model_through_the_driver(monkeypatch):
    posted = {}

    class Reply:
        def raise_for_status(self): pass
        def json(self): return {"message": {"role": "assistant", "content": "hi"}}

    monkeypatch.setattr(egress, "post", lambda url, **kw: (posted.update(kw, url=url), Reply())[1])
    msg = GR._chat([{"role": "user", "content": "x"}])
    assert msg == {"role": "assistant", "content": "hi"}
    assert posted["url"] == driver.OLLAMA_CHAT and posted["json"]["tools"] == GR._TOOLS
    assert posted["json"]["options"] == {"temperature": 0, "num_predict": 2048} and posted["timeout"] == 180


# ---- supplier research ----------------------------------------------------------------------------------

def _research_setup(monkeypatch, gate, turns):
    monkeypatch.setattr(R, "gate_refusal", gate)
    ran = []
    it = iter(turns)
    monkeypatch.setattr(R, "_chat", lambda messages: next(it))
    monkeypatch.setattr(R, "web_search", lambda q: (ran.append(("search", q)), [{"title": "t", "url": "http://a.example", "snippet": "s"}])[1])
    monkeypatch.setattr(R, "fetch_url", lambda u: (ran.append(("fetch", u)), "page text")[1])
    return ran


SEARCH = {"tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "Acme Ltd"}}}]}
FETCH = {"tool_calls": [{"function": {"name": "fetch_url", "arguments": {"url": "http://a.example"}}}]}


def test_a_refused_web_tool_never_runs_and_leaves_no_evidence(monkeypatch):
    ran = _research_setup(monkeypatch, lambda name, args, **k: REFUSAL, [SEARCH, FETCH, {"content": "{}"}])
    content, evidence = R._run_loop("Acme Ltd")
    assert ran == [] and evidence == {}


def test_allowed_web_tools_run_and_record_exactly_the_evidence_they_did_before(monkeypatch):
    ran = _research_setup(monkeypatch, lambda *a, **k: None, [SEARCH, FETCH, {"content": "{}"}])
    content, evidence = R._run_loop("Acme Ltd")
    assert ran == [("search", "Acme Ltd"), ("fetch", "http://a.example")]
    assert evidence == {"http://a.example": "t s page text"}


def test_the_gate_is_asked_per_call_with_the_tool_name_and_arguments(monkeypatch):
    asked = []
    _research_setup(monkeypatch, lambda name, args, **k: asked.append((name, args)), [SEARCH, FETCH, {"content": "{}"}])
    R._run_loop("Acme Ltd")
    assert asked == [("web_search", {"query": "Acme Ltd"}), ("fetch_url", {"url": "http://a.example"})]


def test_supplier_research_reaches_the_model_through_the_driver(monkeypatch):
    posted = {}

    class Reply:
        def raise_for_status(self): pass
        def json(self): return {"message": {"role": "assistant", "content": "{}"}}

    monkeypatch.setattr(egress, "post", lambda url, **kw: (posted.update(kw, url=url), Reply())[1])
    assert R._chat([{"role": "user", "content": "x"}]) == {"role": "assistant", "content": "{}"}
    assert posted["url"] == driver.OLLAMA_CHAT and posted["json"]["tools"] == R._TOOLS


# ---- the ratchet ------------------------------------------------------------------------------------------

def test_every_loop_still_outside_the_executor_asks_the_gate_and_uses_the_driver():
    from tests.orchestration.test_executor import OWN_LOOP_BASELINE
    for rel in OWN_LOOP_BASELINE:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "gate_refusal(" in src, f"{rel} runs a model's tool calls without asking the policy gate"
        assert "agentnick_driver" in src, f"{rel} reaches the model without the driver"


def test_the_gate_these_loops_ask_is_the_real_one_not_a_stand_in():
    assert GR.gate_refusal is tool_runtime.gate_refusal
    assert R.gate_refusal is tool_runtime.gate_refusal
