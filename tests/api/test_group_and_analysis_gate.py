"""Creating an agent group and starting an analysis ask the gate first (RT-23).

Both recorded a write for any signed-in person, or for nobody in particular with sign-in off. Each now
asks the real gate under a write-class name (``agent_group.write``, ``analysis.start``): Buyer and above
proceed, a Viewer is refused and nothing is written, every attempt is audited under its name.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.routers import agent_groups as ag
from api.routers import analysis as an
from src.services import actions, agent_actions, guardrail, rbac
from tests.guardrails.test_guardrail_gate import engine_with
from tests.guardrails.test_rbac import FakePrincipal


def viewer():
    return FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})


def buyer():
    return FakePrincipal("sub-buyer", {"cognito:groups": ["bp-buyers"]})


def test_both_are_writes():
    assert actions.action_class("agent_group.write") == "write"
    assert actions.action_class("analysis.start") == "write"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)
    monkeypatch.setattr(guardrail, "_raise_for_a_human", lambda **k: 1)


@pytest.fixture
def serve(monkeypatch):
    did, audit = [], []
    monkeypatch.setattr(agent_actions, "record_action_or_fail", lambda **k: audit.append(k))
    monkeypatch.setattr(ag.repo, "create", lambda **k: did.append(("group", k.get("created_by"))) or 7)
    monkeypatch.setattr(ag, "_validate", lambda members, links: None)   # slug validation reads the agent catalogue; not what is tested
    monkeypatch.setattr(an.analysis_store, "start", lambda **k: did.append(("analysis", k.get("created_by"))) or "an-1")

    def _serve(router, principal):
        monkeypatch.setattr(rbac, "policy_engine", lambda: engine_with())
        app = FastAPI()
        app.include_router(router.router)
        app.dependency_overrides[router.require_user] = lambda: principal
        client = TestClient(app, raise_server_exceptions=False)
        client.did, client.audit = did, audit
        return client
    return _serve


GROUP_BODY = {"name": "Sourcing pair", "members": [{"agent_slug": "data_extraction"}], "links": []}
CASES = [
    (ag, ag.router.prefix or "/agent-groups", GROUP_BODY, "agent_group.write", "group"),
    (an, (an.router.prefix or "/analysis"), {"session_id": "ses-1", "name": "Q3", "mode": "new"}, "analysis.start", "analysis"),
]
IDS = ["group", "analysis"]


@pytest.mark.parametrize("router,path,body,action,work", CASES, ids=IDS)
def test_a_viewer_is_refused_and_nothing_is_written(serve, router, path, body, action, work):
    client = serve(router, viewer())
    r = client.post(path, json=body)
    assert r.status_code == 403, r.text
    assert client.did == []


@pytest.mark.parametrize("router,path,body,action,work", CASES, ids=IDS)
def test_a_buyer_proceeds_and_is_named(serve, router, path, body, action, work):
    client = serve(router, buyer())
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    assert client.did == [(work, "sub-buyer")]


@pytest.mark.parametrize("router,path,body,action,work", CASES, ids=IDS)
def test_every_attempt_is_audited_under_its_name(serve, router, path, body, action, work):
    refused = serve(router, viewer())
    refused.post(path, json=body)
    assert (action, "denied") in [(a["action_type"], a["status"]) for a in refused.audit]
    allowed = serve(router, buyer())
    allowed.audit.clear()
    allowed.post(path, json=body)
    assert (action, "allowed") in [(a["action_type"], a["status"]) for a in allowed.audit]


@pytest.mark.parametrize("router,path,body,action,work", CASES, ids=IDS)
def test_no_principal_is_refused(serve, router, path, body, action, work):
    """Before this unit a request with no principal wrote a row owned by nobody; the gate now refuses it."""
    client = serve(router, None)
    assert client.post(path, json=body).status_code == 403
    assert client.did == []


def test_changing_or_deleting_a_group_asks_the_same_gate(serve, monkeypatch):
    """Every write to a group is audited under one name; a Viewer (a demoted owner included) is refused
    before the owner-or-Admin rule is even consulted."""
    monkeypatch.setattr(ag.repo, "get", lambda gid: {"group_id": gid, "created_by": "sub-viewer", "members": [], "links": []})
    monkeypatch.setattr(ag.repo, "update", lambda gid, **k: None)
    monkeypatch.setattr(ag.repo, "soft_delete", lambda gid: None)
    client = serve(ag, viewer())
    assert client.put("/agent-groups/5", json={"name": "renamed"}).status_code == 403
    assert client.delete("/agent-groups/5").status_code == 403
    assert [(a["action_type"], a["status"]) for a in client.audit].count(("agent_group.write", "denied")) == 2
