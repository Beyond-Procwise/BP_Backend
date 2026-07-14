# tests/orchestration/test_agent_workflow_repo.py
import pytest

from repositories import agent_workflow_repo as repo
from repositories import workflow_input_request_repo as reqrepo
from orchestration.elicitation import InputRequest
from services.db import get_conn

pytestmark = pytest.mark.integration   # touches the live proc schema

GRAPH = {
    "nodes": [{"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0}],
    "edges": [],
}


def _sweep_namespace():
    """Belt-and-braces: remove anything left behind by this test's namespace,
    including rows orphaned by an earlier crashed/interrupted run."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_agent_workflow WHERE name LIKE 'plan-test%'")
        cur.execute(
            "DELETE FROM proc.bp_workflow_input_request WHERE workflow_id LIKE 'run-plan-test%'"
        )
        cur.close()


@pytest.fixture(scope="module", autouse=True)
def _schema():
    repo.ensure_schema()
    reqrepo.ensure_schema()
    # Sweep before the module runs too, so a prior crashed run never masks a
    # false-positive pass (e.g. list_active() finding a stale row by luck).
    _sweep_namespace()


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    """Hard-delete every row this test creates, on success AND on failure.

    proc.bp_agent_workflow / proc.bp_workflow_input_request live in the
    LIVE production database (bp_sqldb). repo.soft_delete() only flips
    is_active=false — the row stays forever. This fixture guarantees a
    real DELETE runs unconditionally via the post-yield teardown block,
    which pytest executes even when the test body raises.
    """
    created_workflow_ids: list[int] = []
    created_run_ids: list[str] = []

    yield created_workflow_ids, created_run_ids

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            for wid in created_workflow_ids:
                cur.execute(
                    "DELETE FROM proc.bp_agent_workflow WHERE workflow_id = %s", (wid,)
                )
            for run_id in created_run_ids:
                cur.execute(
                    "DELETE FROM proc.bp_workflow_input_request WHERE workflow_id = %s",
                    (run_id,),
                )
            cur.close()
    finally:
        # Belt and braces: sweep by namespace too, so a partially-populated
        # tracking list (e.g. the test failed before appending an id) can't
        # leave orphaned rows behind.
        _sweep_namespace()


def test_create_read_update_soft_delete(_cleanup_test_rows):
    created_workflow_ids, _ = _cleanup_test_rows

    wid = repo.create(name="plan-test", graph=GRAPH, entry_node="n1", created_by="pytest")
    created_workflow_ids.append(wid)
    got = repo.get(wid)
    assert got["name"] == "plan-test"
    assert got["graph"]["nodes"][0]["agent_slug"] == "data_extraction"
    assert got["entry_node"] == "n1"

    repo.update(wid, name="plan-test-renamed")
    assert repo.get(wid)["name"] == "plan-test-renamed"

    assert any(w["workflow_id"] == wid for w in repo.list_active())

    repo.soft_delete(wid)
    assert all(w["workflow_id"] != wid for w in repo.list_active())


def test_input_requests_round_trip(_cleanup_test_rows):
    _, created_run_ids = _cleanup_test_rows

    run_id = "run-plan-test-1"
    created_run_ids.append(run_id)
    reqrepo.raise_requests(run_id, [
        InputRequest(node_id="n1", agent_slug="data_extraction",
                     required_field="s3_prefix", field_type="document_ids",
                     prompt="Which documents should I extract from?")
    ])
    open_ = reqrepo.open_requests(run_id)
    assert len(open_) == 1
    assert open_[0]["required_field"] == "s3_prefix"

    reqrepo.answer(open_[0]["request_id"], "docs/2026/", answered_by="pytest")
    assert reqrepo.open_requests(run_id) == []
    assert reqrepo.answers_for(run_id) == {"s3_prefix": "docs/2026/"}
