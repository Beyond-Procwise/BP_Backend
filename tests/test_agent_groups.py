"""A saved group is a bag of agents you keep using together — not a runnable graph.

The whole reason it has its own table is that proc.bp_agent_workflow's validator
demands a single-entry acyclic DAG, which almost no selection is.
"""
import pytest
from fastapi.testclient import TestClient

# Every group this test suite creates uses one of these names, so teardown can
# find and permanently remove them by name -- including a row orphaned by a
# test that raised before reaching its own client.delete() call. proc.bp_agent_group
# lives in the LIVE bp_sqldb, and the router's DELETE endpoint only soft-deletes
# (is_active=false) -- the row stays in the table forever otherwise. This suite
# hard-deletes by name itself, the same way tests/api/test_agent_workflows_router.py
# hard-deletes proc.bp_agent_workflow rows by name.
_TEST_NAMES = ("Quote triage", "Quote triage (put)")


def _sweep_namespace():
    from services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM proc.bp_agent_group WHERE name = ANY(%s)",
            (list(_TEST_NAMES),),
        )
        cur.close()


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    """Hard-delete every row this suite's namespace could contain, on success
    AND on failure -- the post-yield block runs even when the test body
    raises, so a live-DB row can never survive a failed assertion. Scoped to
    _TEST_NAMES only, so this can never touch a real user's group."""
    yield
    _sweep_namespace()


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


def test_put_with_only_links_is_accepted_against_current_members(client):
    """A PUT that supplies only `links` must not be forced to also resend
    `members` -- and its links are bounds-checked against the group's
    CURRENTLY STORED members, not against an empty list (which the naive
    `body.members or []` implementation rejected unconditionally with
    "A group needs at least one agent.")."""
    body = _group()
    body["name"] = "Quote triage (put)"
    gid = client.post("/agent-groups", json=body).json()["group_id"]

    res = client.put(f"/agent-groups/{gid}", json={"links": []})
    assert res.status_code == 200, res.text

    got = next(g for g in client.get("/agent-groups").json()["groups"] if g["group_id"] == gid)
    assert got["links"] == []
    assert [m["agent_slug"] for m in got["members"]] == ["data_extraction", "supplier_ranking"]

    client.delete(f"/agent-groups/{gid}")


def test_put_with_only_links_out_of_bounds_is_still_rejected(client):
    """The links-only PUT above must not have achieved leniency by skipping
    bounds-checking altogether -- it still validates against the group's
    stored member count (2 members: valid indices are 0 and 1)."""
    body = _group()
    body["name"] = "Quote triage (put)"
    gid = client.post("/agent-groups", json=body).json()["group_id"]

    res = client.put(f"/agent-groups/{gid}", json={"links": [{"from_index": 0, "to_index": 7}]})
    assert res.status_code == 400

    client.delete(f"/agent-groups/{gid}")


def test_never_compiles_a_group(monkeypatch):
    """The guarantee this table exists for."""
    import api.routers.agent_groups as mod
    assert not hasattr(mod, "validate_saved_graph")
    assert "compile_graph" not in dir(mod)
