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
    )


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

    sql_log = []
    app = FastAPI()
    app.include_router(agents_router)
    app.state.agent_nick = _agent_nick(sql_log)
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

    catalogue = json.loads(defs.read_text())["agents"]
    assert any(a["slug"] == "eu-tail-spend-negotiator" for a in catalogue)


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
