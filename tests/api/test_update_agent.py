"""PATCH /agents/{slug} — editing a created agent.

The edit used to rewrite agent_definitions.json. Created agents no longer live
there, so it reported success and changed nothing. It now writes to the store,
and an edited agent goes back to being a draft: what everyone runs has to be
released by someone other than the person who changed it.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import agents.definitions as agent_defs
import api.routers.agents as agents_mod
from api.auth import require_user

BUILTIN = {"agentId": 1, "slug": "negotiation", "agentType": "NegotiationAgent",
           "class_path": "agents.negotiation.NegotiationAgent", "description": "Negotiates."}
CREATED = {"agentId": 9, "slug": "eu-negotiator", "agentType": "NegotiationAgent",
           "class_path": "agents.negotiation.NegotiationAgent", "derived_from": "negotiation",
           "description": "v1", "stage": "published", "created_by": "sub-maker",
           "published_by": "sub-approver"}


class _Store:
    class AgentNotFound(Exception):
        pass

    def __init__(self):
        self.live = {"eu-negotiator": dict(CREATED)}
        self.updates = []

    def update(self, slug, definition, *, by):
        if slug not in self.live:
            raise self.AgentNotFound(slug)
        self.live[slug] = dict(definition)
        self.updates.append((slug, by))
        return len(self.updates) + 1

    def publish(self, slug, definition, *, by):
        self.live[slug] = dict(definition)
        return 99

    def snapshot(self):
        return list(self.live.values()), set()


class _Cursor:
    rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class _Conn:
    def cursor(self):
        return _Cursor()

    def commit(self):
        pass

    def close(self):
        pass


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import json

    defs = tmp_path / "agent_definitions.json"
    defs.write_text(json.dumps({"agents": [BUILTIN]}))
    monkeypatch.setattr(agent_defs, "DEFINITIONS_PATH", defs)
    store = _Store()
    monkeypatch.setattr(agents_mod, "agent_definition_repo", store)
    monkeypatch.setattr(agent_defs, "_created_source", store.snapshot)
    monkeypatch.setattr(agents_mod, "gate", lambda *a, **k: None)
    monkeypatch.setattr(agents_mod, "_do_reload_governance", lambda nick: {})
    monkeypatch.setattr(agents_mod.rbac, "effective_role", lambda p, *a, **k: "Approver")
    monkeypatch.setattr(agents_mod.rbac, "role_rank",
                        lambda role, *a, **k: {"Approver": 3, "Admin": 4}.get(role, 0))
    app = FastAPI()
    app.include_router(agents_mod.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _Conn(), agents={}, auto_registry=None,
        tools=lambda: [SimpleNamespace(name=n, description=n) for n in (
            "get_corpus_facts", "get_policy", "run_email_dispatch")],
    )
    app.state.store = store
    app.state.defs = defs
    return app


def _as(app, subject):
    app.dependency_overrides[require_user] = (
        (lambda: SimpleNamespace(subject=subject)) if subject else (lambda: None))
    return TestClient(app)


def _stored(app):
    return app.state.store.live["eu-negotiator"]


def test_an_edit_is_saved_to_the_store_and_not_to_the_file(app):
    before = app.state.defs.read_text()
    resp = _as(app, "sub-editor").patch("/agents/eu-negotiator", json={"description": "v2"})

    assert resp.status_code == 200, resp.text
    assert _stored(app)["description"] == "v2"
    assert app.state.store.updates == [("eu-negotiator", "sub-editor")]
    assert app.state.defs.read_text() == before
    assert agent_defs.load_agent_definitions()[-1]["description"] == "v2"   # readers see it


def test_only_the_fields_sent_are_changed(app):
    _as(app, "sub-editor").patch("/agents/eu-negotiator", json={"description": "v2"})
    assert _stored(app)["class_path"] == CREATED["class_path"]
    assert _stored(app)["created_by"] == "sub-maker"


def test_an_edited_agent_goes_back_to_draft_as_the_editors(app):
    resp = _as(app, "sub-editor").patch("/agents/eu-negotiator", json={"description": "v2"})

    assert resp.json()["stage"] == "draft"
    stored = _stored(app)
    assert stored["stage"] == "draft"
    assert stored["draft_by"] == "sub-editor"
    assert "published_by" not in stored


def test_the_person_who_edited_it_cannot_publish_it_but_another_approver_can(app):
    _as(app, "sub-editor").patch("/agents/eu-negotiator", json={"description": "v2"})

    assert _as(app, "sub-editor").post("/agents/eu-negotiator/publish").status_code == 403
    # Not even the original creator's earlier approver is barred — only the editor is.
    ok = _as(app, "sub-maker").post("/agents/eu-negotiator/publish")
    assert ok.status_code == 200, ok.text
    assert _stored(app)["stage"] == "published"
    assert "draft_by" not in _stored(app)


def test_with_sign_in_off_an_edit_leaves_the_stage_alone(app):
    resp = _as(app, None).patch("/agents/eu-negotiator", json={"description": "v2"})

    assert resp.status_code == 200, resp.text
    assert _stored(app)["stage"] == "published"
    assert app.state.store.updates == [("eu-negotiator", None)]


def test_a_tool_list_is_checked_and_is_the_agents_tools(app):
    resp = _as(app, "sub-editor").patch("/agents/eu-negotiator", json={"tools": ["get_policy"]})

    assert resp.status_code == 200, resp.text
    assert resp.json()["tools_enforced"] is True
    assert _stored(app)["tools"] == ["get_policy"]


@pytest.mark.parametrize("bad", [["launch_rocket"], ["run_email_dispatch"]])
def test_a_tool_that_cannot_be_given_is_refused_and_nothing_is_saved(app, bad):
    resp = _as(app, "sub-editor").patch("/agents/eu-negotiator", json={"tools": bad})

    assert resp.status_code == 422
    assert app.state.store.updates == []


def test_an_empty_tool_list_is_no_tools_and_null_removes_the_list(app):
    client = _as(app, "sub-editor")
    client.patch("/agents/eu-negotiator", json={"tools": []})
    assert _stored(app)["tools"] == []

    client.patch("/agents/eu-negotiator", json={"tools": None})
    assert "tools" not in _stored(app)


def test_named_outputs_can_be_set_and_removed(app):
    client = _as(app, "sub-editor")
    client.patch("/agents/eu-negotiator",
                 json={"outputs": [{"name": "supplier", "description": "Who."}]})
    assert _stored(app)["output_spec"] == [{"name": "supplier", "description": "Who."}]
    assert _stored(app)["outputs"] == ["supplier"]

    client.patch("/agents/eu-negotiator", json={"outputs": None})
    assert "output_spec" not in _stored(app)
    assert _stored(app)["outputs"] == []


def test_a_bad_output_name_is_refused(app):
    resp = _as(app, "sub-editor").patch("/agents/eu-negotiator",
                                        json={"outputs": [{"name": "Supplier Name"}]})
    assert resp.status_code == 422
    assert app.state.store.updates == []


def test_a_built_in_agent_cannot_be_edited(app):
    assert _as(app, "sub-editor").patch(
        "/agents/negotiation", json={"description": "x"}).status_code == 403


def test_an_agent_that_does_not_exist_is_a_404(app):
    assert _as(app, "sub-editor").patch("/agents/nope", json={"description": "x"}).status_code == 404


def test_the_tool_that_sends_email_is_not_offered_to_pick(app):
    """Offering it would invite a pick that the create and edit calls refuse."""
    resp = _as(app, "sub-editor").get("/agents/tool-options")
    assert resp.status_code == 200, resp.text
    assert [t["key"] for t in resp.json()["tools"]] == ["get_corpus_facts", "get_policy"]
