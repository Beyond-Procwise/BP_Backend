"""Deleting an export record asks the gate first (RT-24).

DELETE /exports/{id} landed without a gate and the ungated-write ratchet went red. It now asks the
real gate under the write-class name export.delete; the store still deletes only a record the caller
created, so the scope is unchanged. Buyer and above proceed, a Viewer is refused and nothing is
deleted, every attempt is audited under that name.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import exports as r
from src.services import actions, agent_actions, guardrail, rbac
from tests.guardrails.test_guardrail_gate import engine_with
from tests.guardrails.test_rbac import FakePrincipal


def viewer():
    return FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})


def buyer():
    return FakePrincipal("sub-buyer", {"cognito:groups": ["bp-buyers"]})


def test_deleting_an_export_is_a_write():
    assert actions.action_class("export.delete") == "write"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)
    monkeypatch.setattr(guardrail, "_raise_for_a_human", lambda **k: 1)


@pytest.fixture
def serve(monkeypatch):
    did, audit = [], []
    monkeypatch.setattr(agent_actions, "record_action_or_fail", lambda **k: audit.append(k))
    monkeypatch.setattr(r.export_store, "delete", lambda export_id, created_by, **k: did.append((export_id, created_by)) or True)

    def _serve(principal):
        monkeypatch.setattr(rbac, "policy_engine", lambda: engine_with())
        app = FastAPI()
        app.include_router(r.router)
        app.dependency_overrides[r.require_user] = lambda: principal
        client = TestClient(app, raise_server_exceptions=False)
        client.did, client.audit = did, audit
        return client
    return _serve


def test_a_viewer_is_refused_and_nothing_is_deleted(serve):
    client = serve(viewer())
    assert client.delete("/exports/abc").status_code == 403
    assert client.did == []
    assert ("export.delete", "denied") in [(a["action_type"], a["status"]) for a in client.audit]


def test_a_buyer_deletes_only_as_themselves(serve):
    client = serve(buyer())
    assert client.delete("/exports/abc").json() == {"deleted": "abc"}
    assert client.did == [("abc", "sub-buyer")]          # the store scopes the delete to the caller
    allowed = [a for a in client.audit if (a["action_type"], a["status"]) == ("export.delete", "allowed")]
    assert allowed and allowed[0]["details"]["export_id"] == "abc"     # the record is named in the audit row


def test_no_principal_is_refused(serve):
    """With no verifier configured require_user yields None; the gate still refuses a write and audits it."""
    client = serve(None)
    assert client.delete("/exports/abc").status_code == 403
    assert client.did == []
    assert ("export.delete", "denied") in [(a["action_type"], a["status"]) for a in client.audit]
