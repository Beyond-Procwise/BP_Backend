"""POST /agent-workflows/{id}/run — the run must not hold the HTTP request.

Review finding F4 / programme item A3: execution was synchronous in-request, so
a GPU-bound run held the connection for minutes and the canvas froze on
"Running…". Now: a run that outlives a short grace period answers "executing"
immediately, and GET /runs/{run_id} serves live per-node progress — statuses,
summarised results, readable errors — until it completes.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.agent_workflows as awf
from api.routers.agent_workflows import router as awf_router

GRAPH = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction"},
        {"id": "n2", "agent_slug": "supplier_ranking"},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


class _SlowEngine:
    """Finishes only when the test says so — the request must not wait for it."""

    def __init__(self):
        self.mid_run = threading.Event()   # set once node 1 is done
        self.release = threading.Event()   # test sets this to let the run finish

    def execute(self, graph, *, input_data=None, user_id="system",
                workflow_id=None, resume_state=None):
        state = resume_state
        state.node_statuses["n1"] = "completed"
        state.node_results["n1"] = {"summary": "82 fields extracted."}
        state.node_statuses["n2"] = "running"
        self.mid_run.set()
        assert self.release.wait(timeout=10), "test never released the engine"
        state.node_statuses["n2"] = "completed"
        state.node_results["n2"] = {"summary": "3 suppliers ranked."}
        state.status = "completed"
        return state


class _RunStore:
    """The persisted run rows, in memory: status transitions only."""

    def __init__(self):
        self.status = {}
        self.finished = []

    def install(self, monkeypatch):
        monkeypatch.setattr(awf.reqrepo, "answers_for", lambda rid: {})
        monkeypatch.setattr(awf.reqrepo, "open_requests", lambda rid: [])
        monkeypatch.setattr(
            awf.reqrepo, "create_run",
            lambda rid, agent_workflow_id, payload, status: self.status.__setitem__(rid, status),
        )
        def claim(rid):
            if self.status.get(rid) == "pending":
                self.status[rid] = "executing"
                return True
            return False
        monkeypatch.setattr(awf.reqrepo, "claim_for_execution", claim)
        def finish(rid, status):
            self.status[rid] = status
            self.finished.append((rid, status))
        monkeypatch.setattr(awf.reqrepo, "finish_run", finish)
        # The run ROW carries agent_workflow_id (create_run stores it).
        # workflow_id_for answers None here on purpose: it reads the run's
        # elicitation-request rows, and a run that asked no questions has
        # none — progress must not depend on it (live bug, 2026-08-22).
        monkeypatch.setattr(
            awf.reqrepo, "get_run",
            lambda rid: {"status": self.status[rid], "agent_workflow_id": 5}
            if rid in self.status else None,
        )
        monkeypatch.setattr(awf.reqrepo, "workflow_id_for", lambda rid: None)
        monkeypatch.setattr(awf.reqrepo, "payload_for", lambda rid: {})


@pytest.fixture()
def slow_app(monkeypatch):
    engine = _SlowEngine()
    store = _RunStore()
    store.install(monkeypatch)
    wf = {"workflow_id": 5, "name": "Two step", "graph": GRAPH}
    monkeypatch.setattr(awf.repo, "get", lambda wid: wf)
    monkeypatch.setattr(awf, "pending_requests", lambda *a, **k: [])
    monkeypatch.setattr(awf, "compile_graph", lambda name, graph: graph)
    monkeypatch.setattr(awf, "governance_for", lambda slug: None, raising=False)
    monkeypatch.setattr(awf, "_SYNC_GRACE_SECONDS", 0.05, raising=True)

    app = FastAPI()
    app.include_router(awf_router)
    app.state.orchestrator = SimpleNamespace(_workflow_engine=engine)
    return TestClient(app), engine, store


def test_a_long_run_answers_executing_instead_of_holding_the_request(slow_app):
    client, engine, store = slow_app
    t0 = time.monotonic()
    resp = client.post("/agent-workflows/5/run", json={"payload": {}})
    elapsed = time.monotonic() - t0

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "executing"
    assert elapsed < 5, "the request waited for the engine"
    engine.release.set()


def test_progress_is_visible_per_node_while_it_runs(slow_app):
    client, engine, store = slow_app
    run_id = client.post("/agent-workflows/5/run", json={"payload": {}}).json()["run_id"]
    assert engine.mid_run.wait(timeout=10)

    mid = client.get(f"/agent-workflows/runs/{run_id}").json()
    assert mid["status"] == "executing"
    assert mid["node_statuses"]["n1"] == "completed"
    assert mid["node_statuses"]["n2"] == "running"
    # The finished node's result is already readable, mid-run.
    assert mid["node_results"]["n1"]["headline"] == "82 fields extracted."
    assert "n2" not in mid["node_results"]

    engine.release.set()
    deadline = time.monotonic() + 10
    final = mid
    while final["status"] == "executing" and time.monotonic() < deadline:
        time.sleep(0.05)
        final = client.get(f"/agent-workflows/runs/{run_id}").json()

    assert final["status"] == "completed"
    assert final["node_results"]["n2"]["headline"] == "3 suppliers ranked."
    assert store.finished == [(run_id, "completed")]


def test_an_engine_crash_marks_the_run_failed_with_a_readable_error(slow_app, monkeypatch):
    client, engine, store = slow_app

    def boom(*a, **k):
        raise RuntimeError("the model backend went away")
    monkeypatch.setattr(engine, "execute", boom)

    run_id = client.post("/agent-workflows/5/run", json={"payload": {}}).json()["run_id"]
    deadline = time.monotonic() + 10
    body = client.get(f"/agent-workflows/runs/{run_id}").json()
    while body["status"] == "executing" and time.monotonic() < deadline:
        time.sleep(0.05)
        body = client.get(f"/agent-workflows/runs/{run_id}").json()

    assert body["status"] == "failed"
    assert body["errors"], "a failed run must say why"
    assert store.finished == [(run_id, "failed")]


def test_a_run_orphaned_by_a_restart_is_reported_failed(slow_app):
    """A run row saying 'executing' with no live thread in this process means
    the server restarted mid-run. The first poll heals it to failed rather
    than reporting 'executing' forever."""
    client, engine, store = slow_app
    store.status["awf-5-deadbeef"] = "executing"

    body = client.get("/agent-workflows/runs/awf-5-deadbeef").json()
    assert body["status"] == "failed"
    assert any("restart" in e.lower() or "interrupted" in e.lower() for e in body["errors"])
    assert ("awf-5-deadbeef", "failed") in store.finished
