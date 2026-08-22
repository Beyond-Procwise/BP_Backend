"""POST /agent-workflows/{id}/run — what a finished run shows the user.

Review finding F2 / programme item A2: the run response carried statuses and
errors but never what any node produced. The response must now carry a
summarised, output-safety-gated view of each node's result — enough to render
a result card, never the agent's raw payload (context snapshots, plans and
action ids stay in the process).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.agent_workflows as awf
from api.routers.agent_workflows import router as awf_router

GRAPH = {
    "nodes": [{"id": "n1", "agent_slug": "supplier_ranking"}],
    "edges": [],
}

RAW_NODE_RESULT = {
    "summary": "3 suppliers ranked for the fastener category.",
    "ranked_suppliers": [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}],
    "top_supplier": "Alpha Fasteners Ltd",
    "confidence": 0.92,
    # Internals that must never reach the browser:
    "action_id": "act-771",
    "context": {"routing_history": ["n1"], "internal_plan": "step 1 ... step 9"},
    "_debug": {"prompt_tokens": 9000},
}


class _FakeEngine:
    def __init__(self, node_results):
        self._node_results = node_results

    def execute(self, graph, input_data, user_id, workflow_id):
        return SimpleNamespace(
            status="completed",
            node_statuses={"n1": "completed"},
            node_results=self._node_results,
            errors=[],
        )


@pytest.fixture()
def run_app(monkeypatch):
    def build(node_results):
        wf = {"workflow_id": 5, "name": "Ranking", "graph": GRAPH}
        monkeypatch.setattr(awf.repo, "get", lambda wid: wf)
        monkeypatch.setattr(awf.reqrepo, "answers_for", lambda rid: {})
        monkeypatch.setattr(awf.reqrepo, "create_run", lambda *a, **k: None)
        monkeypatch.setattr(awf.reqrepo, "claim_for_execution", lambda rid: True)
        monkeypatch.setattr(awf.reqrepo, "finish_run", lambda *a, **k: None)
        monkeypatch.setattr(awf, "pending_requests", lambda *a, **k: [])
        monkeypatch.setattr(awf, "compile_graph", lambda name, graph: graph)
        monkeypatch.setattr(awf, "governance_for", lambda slug: None, raising=False)

        app = FastAPI()
        app.include_router(awf_router)
        app.state.orchestrator = SimpleNamespace(_workflow_engine=_FakeEngine(node_results))
        return TestClient(app)

    return build


def test_run_response_carries_a_summarised_result_per_node(run_app):
    client = run_app({"n1": RAW_NODE_RESULT})
    resp = client.post("/agent-workflows/5/run", json={"payload": {}})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    results = body["node_results"]
    assert "n1" in results
    card = results["n1"]

    assert card["agent_slug"] == "supplier_ranking"
    assert card["headline"] == "3 suppliers ranked for the fastener category."
    # A list field is summarised as a count, never shipped row by row.
    facts = {f["label"]: f["value"] for f in card["facts"]}
    assert facts.get("Ranked suppliers") == "3 items"
    assert facts.get("Top supplier") == "Alpha Fasteners Ltd"


def test_raw_internals_never_reach_the_browser(run_app):
    client = run_app({"n1": RAW_NODE_RESULT})
    resp = client.post("/agent-workflows/5/run", json={"payload": {}})

    text = json.dumps(resp.json())
    assert "act-771" not in text          # action id
    assert "internal_plan" not in text    # context snapshot
    assert "routing_history" not in text
    assert "prompt_tokens" not in text    # underscored/debug keys
    assert "Alpha\"" not in text or "ranked_suppliers" not in text  # no raw rows


def test_headline_is_output_safety_gated(run_app):
    leaky = {"n1": {"summary": "Wrote 3 rows to proc.bp_prompt for you."}}
    client = run_app(leaky)
    resp = client.post("/agent-workflows/5/run", json={"payload": {}})

    card = resp.json()["node_results"]["n1"]
    assert "bp_prompt" not in json.dumps(card)


def test_float_facts_are_rounded_for_reading(run_app):
    """44154.802707999974 is a computation artefact, not a figure a person
    reads. Two decimal places; integers stay integers."""
    client = run_app({"n1": {"total_savings": 44154.802707999974, "count": 7}})
    resp = client.post("/agent-workflows/5/run", json={"payload": {}})

    facts = {f["label"]: f["value"] for f in resp.json()["node_results"]["n1"]["facts"]}
    assert facts["Total savings"] == "44154.80"
    assert facts["Count"] == "7"


def test_a_node_with_no_meaningful_fields_still_gets_a_card(run_app):
    client = run_app({"n1": {"context": {"x": 1}, "_scratch": True}})
    resp = client.post("/agent-workflows/5/run", json={"payload": {}})

    card = resp.json()["node_results"]["n1"]
    assert card["headline"]  # never blank — the card must say something
    assert card["facts"] == []
