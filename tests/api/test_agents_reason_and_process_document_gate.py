"""POST /agents/reason and POST /agents/process-document ask the gate before they act (audit RT-2, RT-3).

Both used to need only a signed-in user. /reason hands AgentNick a free-text task and lets it call tools;
/process-document starts the document-extraction workflow on a caller-chosen S3 prefix or key. Each now asks
the gate for the same action as the nearest governed sibling, first, so a refusal does nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.agents as agents_mod
from api.endpoint_gate import NotPermitted


class _Result:
    def to_dict(self):
        return {"answer": "ok", "trace": []}


class _Nick:
    def __init__(self):
        self.reasoned = []

    def reason(self, task, **kwargs):
        self.reasoned.append(task)
        return _Result()


class _Orchestrator:
    def __init__(self):
        self.workflows = []

    def execute_workflow(self, name, payload, **kwargs):
        self.workflows.append((name, payload))
        return {"status": "started"}


@pytest.fixture()
def rig(monkeypatch):
    asked = []
    monkeypatch.setattr(agents_mod, "gate", lambda action, *a, **k: asked.append(action))
    nick, orch = _Nick(), _Orchestrator()
    app = FastAPI()
    app.include_router(agents_mod.router)
    app.dependency_overrides[agents_mod.require_user] = lambda: None
    app.dependency_overrides[agents_mod.get_agent_nick] = lambda: nick
    app.dependency_overrides[agents_mod.get_orchestrator] = lambda: orch
    return TestClient(app), asked, nick, orch, monkeypatch


def test_reason_asks_the_gate_to_run_an_agent(rig):
    http, asked, nick, _orch, _mp = rig
    resp = http.post("/agents/reason", json={"task": "summarise supplier risk"})
    assert resp.status_code == 200
    assert asked == ["agent.run"]
    assert nick.reasoned == ["summarise supplier risk"]


def test_reason_refused_by_the_gate_reasons_about_nothing(rig):
    http, _asked, nick, _orch, mp = rig

    def refuse(*a, **k):
        raise NotPermitted("no")

    mp.setattr(agents_mod, "gate", refuse)
    resp = http.post("/agents/reason", json={"task": "do something"})
    assert resp.status_code == 403
    assert nick.reasoned == []


def test_process_document_asks_the_gate_to_run_a_workflow(rig):
    http, asked, _nick, orch, _mp = rig
    resp = http.post("/agents/process-document", json={"s3_prefix": "documents/invoice/"})
    assert resp.status_code == 200
    assert asked == ["workflow.run"]
    assert orch.workflows == [("document_extraction", {"s3_prefix": "documents/invoice/", "s3_object_key": None})]


def test_process_document_refused_by_the_gate_starts_nothing(rig):
    http, _asked, _nick, orch, mp = rig

    def refuse(*a, **k):
        raise NotPermitted("no")

    mp.setattr(agents_mod, "gate", refuse)
    resp = http.post("/agents/process-document", json={"s3_object_key": "documents/invoice/a.pdf"})
    assert resp.status_code == 403
    assert orch.workflows == []
