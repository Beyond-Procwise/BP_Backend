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
    monkeypatch.setattr(GR.GT, "get_policy",
                        lambda q: {"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"})
    res = GR.govern("rank these suppliers", "supplier_ranking_agent")
    assert res["answer"] == "Ranked suppliers using the governed weights."
    assert res["governance_used"]["policies"] == [{"policy_type": "supplier_ranking", "slug": "weight_allocation_policy"}]
    assert res["rounds"] == 2


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
