"""POST /agents — the create-agent contract the workspace console actually uses.

The console (engine.js submitCreateAgent) posts {name, description, backing_slug}
and deliberately never sends `instructions` — plain-language behaviour is what
the governed prompts are for. The server must accept that payload and derive
the governed prompt from the description. Review finding F1 / programme item A1:
until this contract held, every create from the live console answered 422.
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

from api.routers.agents import router as agents_router


class _FakeCursor:
    """Records executed SQL; answers RETURNING prompt_id."""

    rowcount = 1

    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return (4242,)


class _FakeConn:
    def __init__(self, log):
        self._log = log

    def cursor(self):
        return _FakeCursor(self._log)

    def commit(self):
        pass

    def close(self):
        pass


class _FakeRegistry:
    def refresh_from_json(self):
        pass

    def get_agent(self, slug):
        return SimpleNamespace(governance_slug=slug.replace("-", "_"))


def _agent_nick(sql_log):
    return SimpleNamespace(
        get_db_connection=lambda: _FakeConn(sql_log),
        agents={},
        auto_registry=_FakeRegistry(),
        tools=lambda: [SimpleNamespace(name=n) for n in (
            "get_corpus_facts", "get_policy", "run_supplier_ranking", "run_email_dispatch",
        )],
    )


class _FakeStore:
    """Stands in for repositories.agent_definition_repo: created agents by slug."""

    class AgentExists(Exception):
        pass

    class AgentNotFound(Exception):
        pass

    def __init__(self):
        self.live = {}
        self.deleted = set()
        self.versions = []

    def create(self, slug, definition, *, derived_from, by):
        if slug in self.live:
            raise self.AgentExists(slug)
        self.live[slug] = dict(definition)
        self.deleted.discard(slug)
        self.versions.append({"slug": slug, "change": "created", "changed_by": by})
        return 1

    def delete(self, slug, definition, *, derived_from, by):
        self.live.pop(slug, None)
        self.deleted.add(slug)
        self.versions.append({"slug": slug, "change": "deleted", "changed_by": by})
        return 2

    def history(self, slug):
        return [v for v in reversed(self.versions) if v["slug"] == slug]

    def snapshot(self):
        return list(self.live.values()), set(self.deleted)


@pytest.fixture()
def create_app(tmp_path, monkeypatch):
    """An app whose catalogue is a scratch file with one creatable base."""
    defs = tmp_path / "agent_definitions.json"
    defs.write_text(json.dumps({
        "agents": [{
            "agentId": 7,
            "slug": "negotiation",
            "description": "Negotiates terms with suppliers.",
            "class_path": "agents.negotiation.NegotiationAgent",
            "capabilities": ["negotiate"],
        }]
    }))
    import agents.definitions as agent_defs
    monkeypatch.setattr(agent_defs, "DEFINITIONS_PATH", defs)
    # Who may create an agent is the gate's question and is tested with the gate;
    # these tests are about what a create that is allowed then does.
    import api.routers.agents as agents_mod
    monkeypatch.setattr(agents_mod, "gate", lambda *a, **k: None)
    # Created agents are stored, not written to the catalogue file.
    store = _FakeStore()
    monkeypatch.setattr(agents_mod, "agent_definition_repo", store)
    monkeypatch.setattr(agent_defs, "_created_source", store.snapshot)
    monkeypatch.setattr(
        "repositories.agent_workflow_repo.list_active", lambda: [], raising=False
    )

    sql_log = []
    app = FastAPI()
    app.include_router(agents_router)
    app.state.agent_nick = _agent_nick(sql_log)
    app.state.store = store
    # Nobody signed in, unless a test says who. Stated here rather than left to
    # whatever sign-in mode an earlier test or the environment set up.
    from api.auth import require_user
    app.dependency_overrides[require_user] = lambda: None
    return app, sql_log, defs


def test_console_payload_without_instructions_creates_the_agent(create_app):
    """The exact body the live console sends must succeed, deriving the
    governed prompt from the description."""
    app, sql_log, defs = create_app
    client = TestClient(app)

    resp = client.post("/agents", json={
        "name": "EU Tail-Spend Negotiator",
        "description": "Negotiates renewal terms for EU tail-spend suppliers.",
        "backing_slug": "negotiation",
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "created"
    assert body["slug"] == "eu-tail-spend-negotiator"

    inserts = [(sql, params) for sql, params in sql_log if "INSERT INTO proc.bp_prompt" in sql]
    assert len(inserts) == 1
    _, params = inserts[0]
    assert "Negotiates renewal terms for EU tail-spend suppliers." in params

    assert "eu-tail-spend-negotiator" in app.state.store.live
    # The catalogue FILE is the built-in agents; a created agent is never written to it.
    assert [a["slug"] for a in json.loads(defs.read_text())["agents"]] == ["negotiation"]


def test_explicit_instructions_still_win_over_description(create_app):
    app, sql_log, _ = create_app
    client = TestClient(app)

    resp = client.post("/agents", json={
        "name": "Freight Negotiator",
        "description": "A short description.",
        "backing_slug": "negotiation",
        "instructions": "Always open by asking for the rate card.",
    })

    assert resp.status_code == 200, resp.text
    inserts = [(sql, params) for sql, params in sql_log if "INSERT INTO proc.bp_prompt" in sql]
    assert len(inserts) == 1
    _, params = inserts[0]
    assert "Always open by asking for the rate card." in params
    assert "A short description." not in params


def test_nothing_to_derive_from_is_still_refused(create_app):
    """No instructions AND no description leaves nothing to govern the agent
    with — refused, and the error names both fields so either console or API
    caller knows what to send."""
    app, sql_log, _ = create_app
    client = TestClient(app)

    resp = client.post("/agents", json={
        "name": "Blank Agent",
        "backing_slug": "negotiation",
    })

    assert resp.status_code == 422
    detail = json.dumps(resp.json())
    assert "instructions" in detail and "description" in detail
    assert not sql_log  # nothing written before the refusal


# --- the tools an agent is given at creation ---------------------------------

def _create(client, **extra):
    return client.post("/agents", json={
        "name": "EU Tail-Spend Negotiator",
        "description": "Negotiates renewal terms for EU tail-spend suppliers.",
        "backing_slug": "negotiation",
        **extra,
    })


def _entry(app):
    return app.state.store.live["eu-tail-spend-negotiator"]


def test_a_tool_list_given_at_creation_is_stored_on_the_agent(create_app):
    app, _, defs = create_app
    resp = _create(TestClient(app), tools=["get_policy", "get_corpus_facts"])

    assert resp.status_code == 200, resp.text
    assert resp.json()["tools"] == ["get_policy", "get_corpus_facts"]
    assert _entry(app)["tools"] == ["get_policy", "get_corpus_facts"]


def test_an_empty_tool_list_is_stored_as_no_tools(create_app):
    app, _, defs = create_app
    resp = _create(TestClient(app), tools=[])

    assert resp.status_code == 200, resp.text
    assert _entry(app)["tools"] == []


def test_no_tool_list_stores_no_list(create_app):
    """Absent is "not narrowed", and must not be written down as an empty list."""
    app, _, defs = create_app
    resp = _create(TestClient(app))

    assert resp.status_code == 200, resp.text
    assert "tools" not in _entry(app)
    assert resp.json()["tools"] is None


def test_a_tool_that_does_not_exist_is_refused_and_nothing_is_created(create_app):
    app, sql_log, defs = create_app
    resp = _create(TestClient(app), tools=["get_policy", "launch_rocket"])

    assert resp.status_code == 422
    assert "launch_rocket" in resp.json()["detail"]
    assert not any("INSERT INTO proc.bp_prompt" in sql for sql, _ in sql_log)
    assert app.state.store.live == {}


def test_the_tool_that_sends_email_cannot_be_given(create_app):
    """A list that named it would be stored and then silently not honoured."""
    app, _, defs = create_app
    resp = _create(TestClient(app), tools=["run_email_dispatch"])

    assert resp.status_code == 422
    assert "run_email_dispatch" in resp.json()["detail"]
    assert app.state.store.live == {}


# --- where a created agent is kept, and what is remembered about it ----------

def test_a_second_agent_with_the_same_name_is_refused(create_app):
    app, _, _ = create_app
    client = TestClient(app)
    assert _create(client).status_code == 200

    again = _create(client)
    assert again.status_code == 422
    assert "already exists" in again.json()["detail"]


def test_with_sign_in_off_nobody_is_named_as_the_creator(create_app):
    app, _, _ = create_app
    _create(TestClient(app))
    assert app.state.store.versions[-1] == {
        "slug": "eu-tail-spend-negotiator", "change": "created", "changed_by": None,
    }


def test_the_person_who_created_it_is_recorded(create_app):
    from api.auth import require_user

    app, _, _ = create_app
    app.dependency_overrides[require_user] = lambda: SimpleNamespace(subject="sub-123")
    _create(TestClient(app))

    assert app.state.store.versions[-1]["changed_by"] == "sub-123"
    assert _entry(app)["created_by"] == "sub-123"


def test_deleting_a_created_agent_marks_it_deleted_and_it_leaves_the_catalogue(create_app):
    import agents.definitions as agent_defs

    app, _, _ = create_app
    client = TestClient(app)
    _create(client)

    resp = client.delete("/agents/eu-tail-spend-negotiator")

    assert resp.status_code == 200, resp.text
    assert "eu-tail-spend-negotiator" in app.state.store.deleted
    assert [a["slug"] for a in agent_defs.load_agent_definitions()] == ["negotiation"]


def test_the_history_of_a_created_agent_can_be_read(create_app):
    app, _, _ = create_app
    client = TestClient(app)
    _create(client)
    client.delete("/agents/eu-tail-spend-negotiator")

    resp = client.get("/agents/eu-tail-spend-negotiator/history")

    assert resp.status_code == 200, resp.text
    assert [v["change"] for v in resp.json()["versions"]] == ["deleted", "created"]


def test_a_built_in_agent_has_no_history(create_app):
    app, _, _ = create_app
    assert TestClient(app).get("/agents/negotiation/history").status_code == 404


# --- the outputs an agent is defined to return --------------------------------

OUTPUTS = [{"name": "supplier", "description": "The supplier that overbilled."},
           {"name": "overbilled_amount", "description": "How much above the PO."}]


def test_named_outputs_are_stored_and_replace_the_ones_copied_from_the_base(create_app):
    app, _, _ = create_app
    resp = _create(TestClient(app), outputs=OUTPUTS)

    assert resp.status_code == 200, resp.text
    assert resp.json()["outputs"] == OUTPUTS
    entry = _entry(app)
    assert entry["output_spec"] == OUTPUTS
    assert entry["outputs"] == ["supplier", "overbilled_amount"]


def test_an_agent_created_without_outputs_names_none(create_app):
    app, _, _ = create_app
    resp = _create(TestClient(app))

    assert resp.json()["outputs"] is None
    assert "output_spec" not in _entry(app)


def test_a_bad_output_name_is_refused_and_nothing_is_created(create_app):
    app, sql_log, _ = create_app
    resp = _create(TestClient(app), outputs=[{"name": "Supplier Name"}])

    assert resp.status_code == 422
    assert "Supplier Name" in resp.json()["detail"]
    assert app.state.store.live == {}
    assert not any("INSERT INTO proc.bp_prompt" in sql for sql, _ in sql_log)


def test_a_workflow_publishes_a_created_agents_named_outputs(create_app):
    """What the next step on a canvas can use is what the agent was defined to return."""
    from orchestration.workflow_compiler import compile_graph

    app, _, _ = create_app
    _create(TestClient(app), outputs=OUTPUTS)

    wf = compile_graph("t", {"nodes": [{"id": "n1", "agent_slug": "eu-tail-spend-negotiator"}],
                             "edges": []})
    node = wf.nodes["n1"]
    assert node.tool_loop is True
    assert node.output_to_shared == ["supplier", "overbilled_amount", "answer"]


# --- a new agent is a draft ----------------------------------------------------

def test_an_agent_created_by_a_signed_in_person_starts_as_a_draft(create_app):
    from api.auth import require_user

    app, _, _ = create_app
    app.dependency_overrides[require_user] = lambda: SimpleNamespace(subject="sub-123")
    resp = _create(TestClient(app))

    assert resp.json()["stage"] == "draft"
    assert _entry(app)["stage"] == "draft"


def test_with_sign_in_off_there_is_nobody_to_keep_a_draft_to(create_app):
    """No creator to restrict it to and no second person to release it: it is
    published at once, as every agent was before drafts existed."""
    app, _, _ = create_app
    resp = _create(TestClient(app))

    assert resp.json()["stage"] == "published"
    assert _entry(app)["stage"] == "published"
