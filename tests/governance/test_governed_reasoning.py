"""Governance tools + AgentNick governed-reasoning loop + endpoint."""
import pytest

from src.services.db import get_conn
from src.services.governance_tools import governed_reasoning as GR
from src.services.governance_tools import tools as GT


def _db_or_skip():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")


def test_tools_match_governance():
    _db_or_skip()
    GT.refresh()
    assert GT.get_policy("supplier_ranking").get("policy_type") == "supplier_ranking"
    assert GT.get_prompt("supplier_ranking_agent").get("prompt_name")
    lg = GT.list_governance("negotiation_agent")
    assert any(p["prompt_name"] == "negotiation_message_default" for p in lg["prompts"])


def test_govern_loop_records_governance(monkeypatch):
    calls = {"n": 0}

    def fake_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_calls": [{"function": {"name": "get_policy", "arguments": {"query": "supplier_ranking"}}}]}
        return {"content": "Ranked suppliers using the governed weights."}

    monkeypatch.setattr(GR, "_chat", fake_chat)
    monkeypatch.setattr(GR.GT, "refresh", lambda: None)
    monkeypatch.setattr(GR.GT, "list_governance", lambda a=None: {"policies": [], "prompts": []})
    monkeypatch.setattr(GR.GT, "get_policy",
                        lambda q: {"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"})
    res = GR.govern("rank these suppliers", "supplier_ranking_agent")
    assert res["answer"] == "Ranked suppliers using the governed weights."
    assert res["governance_used"]["policies"] == [{"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"}]
    assert res["rounds"] == 2


def test_govern_flags_cited_but_unfetched_prompt(monkeypatch):
    """A prompt the model cites but never fetched must land in `unsupported`,
    not be passed off inside `governance_used`."""
    calls = {"n": 0}

    def fake_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_calls": [{"function": {"name": "get_policy", "arguments": {"query": "supplier_ranking"}}}]}
        # Answers naming a prompt it never fetched, and declares it in CITED.
        return {"content": (
            "Ranked using the governed weights and the supplier_ranking_justification prompt.\n"
            'CITED: {"policies": ["weight_allocation_policy"], "prompts": ["supplier_ranking_justification"]}'
        )}

    monkeypatch.setattr(GR, "_chat", fake_chat)
    monkeypatch.setattr(GR.GT, "refresh", lambda: None)
    monkeypatch.setattr(GR.GT, "list_governance", lambda a=None: {"policies": [], "prompts": []})
    monkeypatch.setattr(GR.GT, "get_policy",
                        lambda q: {"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"})
    res = GR.govern("rank these suppliers", "supplier_ranking_agent")
    # policy was fetched -> supported; prompt was never fetched -> unsupported
    assert res["unsupported"]["prompts"] == ["supplier_ranking_justification"]
    assert res["unsupported"]["policies"] == []
    assert res["governance_used"]["prompts"] == []
    # CITED line is stripped from the prose answer
    assert "CITED:" not in res["answer"]
    assert "supplier_ranking_justification" in res["answer"]


def test_govern_flags_prose_fabrication_with_clean_cited(monkeypatch):
    """Prose names a prompt it never fetched while CITED stays honestly clean.

    The CITED cross-check can't catch this; the catalog prose-scan must."""
    calls = {"n": 0}

    def fake_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_calls": [{"function": {"name": "get_policy", "arguments": {"query": "supplier_ranking"}}}]}
        return {"content": (
            "Ranking uses the weight allocation policy and, for explanations, the "
            "supplier_ranking_justification prompt.\n"
            'CITED: {"policies": ["weight_allocation_policy"], "prompts": []}'
        )}

    monkeypatch.setattr(GR, "_chat", fake_chat)
    monkeypatch.setattr(GR.GT, "refresh", lambda: None)
    monkeypatch.setattr(GR.GT, "get_policy",
                        lambda q: {"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"})
    # Catalog: both governance names EXIST; only the policy was fetched above.
    monkeypatch.setattr(GR.GT, "list_governance", lambda a=None: {
        "policies": [{"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"}],
        "prompts": [{"prompt_name": "supplier_ranking_justification", "prompt_type": "supplier_ranking"}],
    })
    res = GR.govern("rank", "supplier_ranking_agent")
    # prose names an unfetched prompt (CITED omitted it) -> caught by prose scan
    assert res["unsupported"]["prompts"] == ["supplier_ranking_justification"]
    # the fetched policy, though named in prose, is NOT flagged
    assert res["unsupported"]["policies"] == []


def test_govern_passes_fetched_citation(monkeypatch):
    """A citation the model actually fetched stays supported; unsupported empty."""
    calls = {"n": 0}

    def fake_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool_calls": [{"function": {"name": "get_policy", "arguments": {"query": "supplier_ranking"}}}]}
        return {"content": 'Done.\nCITED: {"policies": ["weight_allocation_policy"], "prompts": []}'}

    monkeypatch.setattr(GR, "_chat", fake_chat)
    monkeypatch.setattr(GR.GT, "refresh", lambda: None)
    monkeypatch.setattr(GR.GT, "list_governance", lambda a=None: {"policies": [], "prompts": []})
    monkeypatch.setattr(GR.GT, "get_policy",
                        lambda q: {"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"})
    res = GR.govern("rank", "supplier_ranking_agent")
    assert res["unsupported"] == {"policies": [], "prompts": []}
    assert res["governance_used"]["policies"] == [{"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"}]


def test_govern_corrective_retry_when_no_tools_used(monkeypatch):
    """Answering with zero tool calls triggers exactly one 'consult governance' nudge."""
    calls = {"n": 0}
    seen_nudge = {"v": False}

    def fake_chat(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            # tries to answer immediately, no tools
            return {"content": "The weights are risk 0.2, price 0.4."}
        if calls["n"] == 2:
            # after nudge, the last message should be the corrective user turn
            seen_nudge["v"] = messages[-1]["role"] == "user"
            return {"tool_calls": [{"function": {"name": "get_policy", "arguments": {"query": "supplier_ranking"}}}]}
        return {"content": 'Governed answer.\nCITED: {"policies": ["weight_allocation_policy"], "prompts": []}'}

    monkeypatch.setattr(GR, "_chat", fake_chat)
    monkeypatch.setattr(GR.GT, "refresh", lambda: None)
    monkeypatch.setattr(GR.GT, "list_governance", lambda a=None: {"policies": [], "prompts": []})
    monkeypatch.setattr(GR.GT, "get_policy",
                        lambda q: {"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"})
    res = GR.govern("what weights?", "supplier_ranking_agent")
    assert seen_nudge["v"], "a corrective user turn should precede the retry"
    assert res["answer"] == "Governed answer."
    assert res["governance_used"]["policies"] == [{"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"}]
    assert res["rounds"] == 3


def test_govern_endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routers import governance as G
    import src.services.governance_tools.governed_reasoning as GRmod

    monkeypatch.setattr(GRmod, "govern",
                        lambda task, agent=None: {"answer": "ok", "governance_used": {"prompts": [], "policies": []}, "rounds": 1})
    app = FastAPI(); app.include_router(G.router)
    client = TestClient(app)
    assert client.post("/agents/govern", json={"task": "x", "agent": "a"}).json()["answer"] == "ok"
    assert client.post("/agents/govern", json={"task": "  "}).status_code == 400
