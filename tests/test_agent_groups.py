"""A saved group is a bag of agents you keep using together — not a runnable graph.

The whole reason it has its own table is that proc.bp_agent_workflow's validator
demands a single-entry acyclic DAG, which almost no selection is.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    # api/main.py ensures proc.bp_agent_group exists in its lifespan startup
    # (see api/main.py's lifespan, beside _agent_workflow_repo.ensure_schema()).
    # TestClient only runs that startup when used as a context manager -- a
    # bare `TestClient(app)` never fires it, so this table would not exist
    # yet on a fresh run against the live database, matching the pattern in
    # tests/api/test_agent_workflows_router.py.
    with TestClient(app) as c:
        yield c


def _group():
    return {
        "name": "Quote triage",
        "members": [
            {"agent_slug": "data_extraction", "display_name": "Data Extraction", "dx": 0, "dy": 0},
            {"agent_slug": "supplier_ranking", "display_name": "Supplier Ranking", "dx": 150, "dy": 0},
        ],
        "links": [{"from_index": 0, "to_index": 1}],
    }


def test_round_trips_a_group(client):
    created = client.post("/agent-groups", json=_group())
    assert created.status_code == 200, created.text
    gid = created.json()["group_id"]

    rows = client.get("/agent-groups").json()["groups"]
    mine = next(g for g in rows if g["group_id"] == gid)
    assert mine["name"] == "Quote triage"
    assert [m["agent_slug"] for m in mine["members"]] == ["data_extraction", "supplier_ranking"]
    assert mine["links"] == [{"from_index": 0, "to_index": 1}]

    client.delete(f"/agent-groups/{gid}")
    assert all(g["group_id"] != gid for g in client.get("/agent-groups").json()["groups"])


def test_accepts_a_selection_no_workflow_would(client):
    """Two disconnected agents: two entry nodes, which validate_saved_graph rejects.
    A group is stored, never compiled, so this must be accepted."""
    body = _group()
    body["links"] = []
    res = client.post("/agent-groups", json=body)
    assert res.status_code == 200, res.text
    client.delete(f"/agent-groups/{res.json()['group_id']}")


def test_rejects_an_agent_that_does_not_exist(client):
    body = _group()
    body["members"][0]["agent_slug"] = "not_a_real_agent"
    res = client.post("/agent-groups", json=body)
    assert res.status_code == 400
    assert "not_a_real_agent" in res.text


def test_rejects_a_link_pointing_outside_the_group(client):
    body = _group()
    body["links"] = [{"from_index": 0, "to_index": 7}]
    res = client.post("/agent-groups", json=body)
    assert res.status_code == 400


def test_never_compiles_a_group(monkeypatch):
    """The guarantee this table exists for."""
    import api.routers.agent_groups as mod
    assert not hasattr(mod, "validate_saved_graph")
    assert "compile_graph" not in dir(mod)
