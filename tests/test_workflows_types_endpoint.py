"""Regression test for GET /workflows/types.

Background: the production handler used ``[AgentType(**agent) for agent in data]``
where ``data = json.load(open("agent_definitions.json"))`` returns a dict
``{"agents": [...]}``.  Iterating a dict yields its keys (strings), so the call
became ``AgentType(**"agents")`` which raises
``TypeError: argument after ** must be a mapping, not str`` and the route
re-raised it as HTTP 500.
"""

import json
import os
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers.workflows import router as workflows_router  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(workflows_router)
    return TestClient(app)


def test_get_agent_types_returns_list_from_real_definitions_file():
    response = _client().get("/workflows/types")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert body, "expected at least one agent in agent_definitions.json"
    first = body[0]
    for required in ("agentId", "agentType", "description", "dependencies"):
        assert required in first, f"missing field {required!r}: {first}"


def test_get_agent_types_count_matches_definitions_file():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(repo_root, "agent_definitions.json")) as f:
        defs = json.load(f)
    response = _client().get("/workflows/types")
    assert response.status_code == 200
    assert len(response.json()) == len(defs["agents"])
