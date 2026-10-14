"""Reading the record of a created agent's run."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.agents as agents_mod
from repositories import agent_run_repo

RUN = {"run_id": "abc123", "agent_slug": "overcharge-hunter", "model": "m",
       "system_prompt": "You are the overcharge hunter.", "calls": [], "answer": "ok"}


@pytest.fixture()
def client(monkeypatch):
    asked = []
    monkeypatch.setattr(agents_mod, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(agent_run_repo, "get", lambda run_id: RUN if run_id == "abc123" else None)
    monkeypatch.setattr(agent_run_repo, "list_for_agent",
                        lambda slug, limit=50: [{"run_id": "abc123", "agent_slug": slug}])
    app = FastAPI()
    app.include_router(agents_mod.router)
    # Who is signed in is not what these tests are about; the gate is.
    from api.auth import require_user
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app), asked


def test_a_recorded_run_can_be_read(client):
    http, asked = client
    resp = http.get("/agents/runs/abc123")

    assert resp.status_code == 200
    assert resp.json()["system_prompt"] == "You are the overcharge hunter."
    assert asked == ["agent.run"]           # the same question as running the agent


def test_a_run_that_was_never_recorded_is_a_404(client):
    http, _ = client
    assert http.get("/agents/runs/nope").status_code == 404


def test_an_agents_runs_are_listed(client):
    http, asked = client
    resp = http.get("/agents/overcharge-hunter/runs")

    assert resp.status_code == 200
    assert resp.json() == {"slug": "overcharge-hunter",
                           "runs": [{"run_id": "abc123", "agent_slug": "overcharge-hunter"}]}
    assert asked == ["agent.run"]


def test_someone_refused_at_the_gate_reads_nothing(client, monkeypatch):
    http, _ = client

    def refuse(*a, **k):
        raise HTTPException(status_code=403, detail="not permitted")

    monkeypatch.setattr(agents_mod, "gate", refuse)
    assert http.get("/agents/runs/abc123").status_code == 403
    assert http.get("/agents/overcharge-hunter/runs").status_code == 403
