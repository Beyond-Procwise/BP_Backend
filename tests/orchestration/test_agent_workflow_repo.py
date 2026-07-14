# tests/orchestration/test_agent_workflow_repo.py
import pytest

from repositories import agent_workflow_repo as repo
from repositories import workflow_input_request_repo as reqrepo
from orchestration.elicitation import InputRequest

pytestmark = pytest.mark.integration   # touches the live proc schema

GRAPH = {
    "nodes": [{"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0}],
    "edges": [],
}


@pytest.fixture(scope="module", autouse=True)
def _schema():
    repo.ensure_schema()
    reqrepo.ensure_schema()


def test_create_read_update_soft_delete():
    wid = repo.create(name="plan-test", graph=GRAPH, entry_node="n1", created_by="pytest")
    got = repo.get(wid)
    assert got["name"] == "plan-test"
    assert got["graph"]["nodes"][0]["agent_slug"] == "data_extraction"
    assert got["entry_node"] == "n1"

    repo.update(wid, name="plan-test-renamed")
    assert repo.get(wid)["name"] == "plan-test-renamed"

    assert any(w["workflow_id"] == wid for w in repo.list_active())

    repo.soft_delete(wid)
    assert all(w["workflow_id"] != wid for w in repo.list_active())


def test_input_requests_round_trip():
    run_id = "run-plan-test-1"
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
