"""Publishing a draft agent, and reading what agents have cost."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import agents.definitions as agent_defs
import api.routers.agents as agents_mod
from api.auth import require_user
from repositories import agent_run_repo

DRAFT = {"agentId": 9, "slug": "eu-negotiator", "derived_from": "negotiation",
         "class_path": "agents.negotiation.NegotiationAgent", "stage": "draft",
         "created_by": "sub-maker"}


class _Store:
    class AgentNotFound(Exception):
        pass

    def __init__(self):
        self.live = {"eu-negotiator": dict(DRAFT)}
        self.calls = []

    def publish(self, slug, definition, *, by):
        if slug not in self.live:
            raise self.AgentNotFound(slug)
        self.live[slug] = dict(definition)
        self.calls.append((slug, by))
        return 2

    def snapshot(self):
        return list(self.live.values()), set()


@pytest.fixture()
def app(monkeypatch):
    store = _Store()
    monkeypatch.setattr(agents_mod, "agent_definition_repo", store)
    monkeypatch.setattr(agent_defs, "_created_source", store.snapshot)
    roles = {"sub-maker": "Approver", "sub-approver": "Approver", "sub-buyer": "Buyer",
             "sub-admin": "Admin"}
    monkeypatch.setattr(agents_mod.rbac, "effective_role",
                        lambda principal, *a, **k: roles.get(getattr(principal, "subject", None),
                                                             "Viewer"))
    ranks = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}
    monkeypatch.setattr(agents_mod.rbac, "role_rank", lambda role, *a, **k: ranks.get(role, 0))
    app = FastAPI()
    app.include_router(agents_mod.router)
    app.state.agent_nick = SimpleNamespace(agents={}, auto_registry=None)
    app.state.store = store
    return app


def _as(app, subject):
    app.dependency_overrides[require_user] = (
        (lambda: SimpleNamespace(subject=subject)) if subject else (lambda: None))
    return TestClient(app)


def test_a_second_person_with_approval_rights_publishes_a_draft(app):
    resp = _as(app, "sub-approver").post("/agents/eu-negotiator/publish")

    assert resp.status_code == 200, resp.text
    assert resp.json()["stage"] == "published"
    stored = app.state.store.live["eu-negotiator"]
    assert stored["stage"] == "published"
    assert stored["published_by"] == "sub-approver"
    assert app.state.store.calls == [("eu-negotiator", "sub-approver")]


def test_the_person_who_created_it_cannot_publish_it(app):
    resp = _as(app, "sub-maker").post("/agents/eu-negotiator/publish")

    assert resp.status_code == 403
    assert "created" in resp.json()["detail"]
    assert app.state.store.live["eu-negotiator"]["stage"] == "draft"


def test_someone_without_approval_rights_cannot_publish(app):
    assert _as(app, "sub-buyer").post("/agents/eu-negotiator/publish").status_code == 403
    assert app.state.store.live["eu-negotiator"]["stage"] == "draft"


def test_nobody_signed_in_cannot_publish(app):
    assert _as(app, None).post("/agents/eu-negotiator/publish").status_code == 403


def test_an_agent_that_is_not_a_draft_cannot_be_published(app):
    app.state.store.live["eu-negotiator"]["stage"] = "published"
    assert _as(app, "sub-approver").post("/agents/eu-negotiator/publish").status_code == 409


def test_an_agent_that_does_not_exist_is_a_404(app):
    assert _as(app, "sub-approver").post("/agents/nope/publish").status_code == 404


# --- usage ---------------------------------------------------------------------

@pytest.fixture()
def usage(monkeypatch):
    asked = {}

    def fake(since, until):
        asked.update(since=since, until=until)
        return [{"agent_slug": "eu-negotiator", "model": "m", "user_id": "sub-maker",
                 "runs": 2, "model_calls": 5, "prompt_tokens": 3000, "completion_tokens": 200}]

    monkeypatch.setattr(agent_run_repo, "usage_summary", fake)
    return asked


def test_an_admin_reads_usage_for_a_period(app, usage):
    resp = _as(app, "sub-admin").get("/agents/usage?since=2026-10-01&until=2026-11-01")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rows"][0]["prompt_tokens"] == 3000
    assert body["totals"] == {"runs": 2, "model_calls": 5, "prompt_tokens": 3000,
                              "completion_tokens": 200}
    assert usage["since"] == datetime(2026, 10, 1) and usage["until"] == datetime(2026, 11, 1)


@pytest.mark.parametrize("subject", ["sub-approver", "sub-buyer", None])
def test_nobody_below_admin_reads_usage(app, usage, subject):
    assert _as(app, subject).get(
        "/agents/usage?since=2026-10-01&until=2026-11-01").status_code == 403
    assert usage == {}


def test_a_period_that_ends_before_it_starts_is_refused(app, usage):
    assert _as(app, "sub-admin").get(
        "/agents/usage?since=2026-11-01&until=2026-10-01").status_code == 422
