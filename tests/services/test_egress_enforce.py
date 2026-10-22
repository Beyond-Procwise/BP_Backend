"""1.20 step (d): enforcement, one purpose at a time, switched by the administrator's policy row.

The EgressPolicy row's ``enforce`` list names the purposes whose shadow decision is now APPLIED:
a call of an enforced purpose that the rules deny is refused before it leaves (request() returns
None and records ``denied``; observe() raises EgressDenied). Everything else stays shadow. Local
destinations are never refused. The list is empty in the seed: nothing is enforced until an Admin
says so after the shadow week (D-62).

An unreadable row never widens enforcement: the last successfully read LIST of enforced purposes
stays in force; a process that has never read the row enforces nothing. The VERDICT in an outage is
still "deny" for every external destination (fail closed), so an enforced purpose refuses all its
external calls until the row can be read again; the last-good rules do not decide in its place.
"""
from __future__ import annotations

import json

import pytest

from src.services import egress, egress_log
from src.services import egress_policy as ep
from src.services.egress import EgressDenied, Purpose

RULES = {
    "allow": [{"purpose": "supplier_research", "hosts": ["html.duckduckgo.com"], "agents": ["*"]},
              {"purpose": "fx_rates", "hosts": ["open.er-api.com"], "agents": ["*"]}],
    "deny": [{"purpose": "model_inference", "hosts": ["api.ollama.com"], "agents": ["*"]}],
    "enforce": ["supplier_research", "model_inference"],
}


class _Resp:
    status_code = 200
    headers = {}
    text = ""


@pytest.fixture
def world(monkeypatch):
    calls, events = [], []
    monkeypatch.setattr(egress.requests, "request", lambda m, u, **kw: calls.append((m, u)) or _Resp())
    monkeypatch.setattr(egress, "_resolve",
                        lambda host: ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"])
    monkeypatch.setattr(egress_log, "record", lambda **kw: events.append(kw))
    monkeypatch.setattr(ep, "_load_row", lambda: ep.Policy(version=3, rules=RULES))
    ep.invalidate()
    yield calls, events
    ep.invalidate(forget_last_good=True)      # process-wide: must not leak enforcement into other tests


# ---- the row ---------------------------------------------------------------------------------------

def test_validate_accepts_an_enforce_list_of_known_purposes_and_refuses_others():
    out = ep.validate({"allow": [], "deny": [], "enforce": ["fx_rates", "fx_rates"]})
    assert out["enforce"] == ["fx_rates"]
    assert ep.validate({"allow": []})["enforce"] == []
    for bad in ({"enforce": ["teleport"]}, {"enforce": "fx_rates"}, {"enforce": [1]}):
        with pytest.raises(ep.InvalidRules):
            ep.validate(bad)


def test_the_seed_enforces_nothing():
    assert ep.SEED_RULES["enforce"] == []
    assert ep.Policy(version=1, rules=ep.SEED_RULES).enforced(Purpose.SUPPLIER_RESEARCH) is False


def test_enforced_for_reads_the_row_and_keeps_the_last_known_list_through_an_outage(monkeypatch):
    monkeypatch.setattr(ep, "_load_row", lambda: ep.Policy(version=3, rules=RULES))
    ep.invalidate()
    assert ep.enforced_for(Purpose.SUPPLIER_RESEARCH) is True
    assert ep.enforced_for(Purpose.FX_RATES) is False

    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(ep, "_load_row", boom)
    ep.invalidate()
    assert ep.enforced_for(Purpose.SUPPLIER_RESEARCH) is True, "the last good list stays in force"
    assert ep.enforced_for(Purpose.FX_RATES) is False, "an outage widens nothing"
    ep.invalidate(forget_last_good=True)
    assert ep.enforced_for(Purpose.SUPPLIER_RESEARCH) is False, "never read: nothing is enforced"


# ---- request() -------------------------------------------------------------------------------------

def test_an_enforced_purpose_that_the_rules_deny_is_refused_before_it_leaves(world):
    calls, events = world
    assert egress.get("https://evil.example/", purpose=Purpose.SUPPLIER_RESEARCH, agent="r") is None
    assert calls == []
    e = events[-1]
    assert e["outcome"] == "denied" and e["shadow_decision"] == "deny" and e["policy_version"] == "egress-policy-v3"
    assert "evil.example" in e["detail"]


def test_an_enforced_purpose_that_the_rules_allow_still_goes_out(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "capacity", lambda: (True, ""))   # the real writer thread, with no database, must not decide this
    assert egress.get("https://html.duckduckgo.com/html/?q=Borcelle", purpose=Purpose.SUPPLIER_RESEARCH) is not None
    assert len(calls) == 1 and events[-1]["outcome"] == "http_200"


def test_the_classification_cap_is_enforced_too(world):
    calls, events = world
    assert egress.post("https://html.duckduckgo.com/html/", purpose=Purpose.SUPPLIER_RESEARCH,
                       json={"q": "Borcelle INV-2024-001 £17,065.50"}) is None
    assert calls == [] and "commercial" in events[-1]["detail"]


def test_a_purpose_not_in_the_enforce_list_stays_shadow(world):
    calls, events = world
    assert egress.get("https://evil.example/", purpose=Purpose.FX_RATES) is not None
    assert len(calls) == 1 and events[-1]["shadow_decision"] == "deny" and events[-1]["outcome"] == "http_200"


def test_a_local_destination_is_never_refused_even_when_its_purpose_is_enforced(world):
    calls, events = world
    assert egress.post("http://localhost:11434/api/generate", purpose=Purpose.MODEL_INFERENCE,
                       require_global=False, json={"prompt": "x"}) is not None
    assert len(calls) == 1


def test_ollama_cloud_is_refused_once_model_inference_is_enforced(world):
    calls, events = world
    assert egress.post("https://api.ollama.com/api/generate", purpose=Purpose.MODEL_INFERENCE,
                       json={"prompt": "company text"}) is None
    assert calls == [] and events[-1]["outcome"] == "denied"


def test_request_or_raise_names_the_refusal(world):
    with pytest.raises(EgressDenied):
        egress.request_or_raise("GET", "https://evil.example/", purpose=Purpose.SUPPLIER_RESEARCH)


# ---- observe() ---------------------------------------------------------------------------------------

def test_an_observed_library_call_is_refused_before_the_library_runs(world):
    calls, events = world
    ran = []
    with pytest.raises(EgressDenied) as exc:
        with egress.observe(Purpose.SUPPLIER_RESEARCH, "evil.example", agent="r", payload="q"):
            ran.append(1)
    assert ran == [] and events[-1]["outcome"] == "denied" and "evil.example" in exc.value.reason


# ---- the research loop tells the model why ---------------------------------------------------------

def test_the_research_loop_hands_the_model_a_readable_refusal(monkeypatch):
    from src.services.supplier_enrichment import research as R

    turns = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "web_search", "arguments": {"query": "Borcelle Ltd"}}}]},
        {"role": "assistant", "content": '{"matched_name": null}'},
    ]
    seen = []
    monkeypatch.setattr(R, "_chat", lambda messages: seen.append(list(messages)) or turns.pop(0))
    monkeypatch.setattr(R, "_MAX_ROUNDS", lambda: 3)
    monkeypatch.setattr(R, "gate_refusal", lambda *a, **k: None)

    def refuse(query, max_results=5):
        raise EgressDenied(reason="no rule allows supplier_research to html.duckduckgo.com",
                           destination="html.duckduckgo.com", purpose="supplier_research")
    monkeypatch.setattr(R, "web_search", refuse)
    content, evidence = R._run_loop("Borcelle Ltd")
    assert content == '{"matched_name": null}' and evidence == {}
    tool_msg = [m for m in seen[-1] if m.get("role") == "tool"][-1]
    told = json.loads(tool_msg["content"])
    assert told["result"] == "blocked" and told["reasonCode"] == "egress_policy"
    assert told["reason"] == "The outside-research policy does not allow this search."
    for word in ("html.duckduckgo.com", "rule for", "permits up to", "commercial", "public", "agent "):
        assert word not in told["reason"], "the verdict would tell the model what to strip and retry"


def test_web_search_re_raises_an_egress_refusal(monkeypatch):
    import sys
    import types
    from src.services.supplier_enrichment import web_tools
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=object))

    def refuse(*a, **k):
        raise EgressDenied(reason="policy", destination="html.duckduckgo.com", purpose="supplier_research")
    monkeypatch.setattr(web_tools, "egress_observe", refuse)
    with pytest.raises(EgressDenied):
        web_tools.web_search("Borcelle")
