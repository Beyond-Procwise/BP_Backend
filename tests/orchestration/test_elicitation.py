from orchestration.elicitation import InputRequest, pending_requests

EXTRACT_THEN_RANK = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction"},
        {"id": "n2", "agent_slug": "supplier_ranking"},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


def test_data_extraction_with_no_upstream_asks_for_documents():
    """THE regression guard. The old manifests would have asked for nothing here,
    and the agent would have run against no documents and produced nothing."""
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={})
    doc = [r for r in reqs if r.node_id == "n1"]
    assert len(doc) == 1
    assert doc[0].agent_slug == "data_extraction"
    assert doc[0].field_type == "document_ids"
    assert doc[0].required_field == "s3_object_keys"   # first member of the any_of group


def test_supplier_ranking_always_asks_for_a_query():
    """Nothing in the system produces `query`, so it must come from the human."""
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={})
    assert any(r.node_id == "n2" and r.field_type == "text" for r in reqs)


def test_a_group_satisfied_by_the_run_payload_is_not_asked_for():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={"s3_prefix": "docs/"}, answers={})
    assert [r.node_id for r in reqs] == ["n2"]      # only the ranking query remains


def test_any_member_of_the_group_satisfies_it():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={"document_ids": [1, 2]}, answers={})
    assert not [r for r in reqs if r.node_id == "n1"]


def test_a_prior_human_answer_satisfies_a_group():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={"s3_prefix": "docs/"})
    assert not [r for r in reqs if r.node_id == "n1"]


def test_an_upstream_output_satisfies_a_downstream_group():
    """discrepancy_detection consumes `details`, which data_extraction produces —
    so it must NOT be asked for."""
    graph = {
        "nodes": [
            {"id": "n1", "agent_slug": "data_extraction"},
            {"id": "n2", "agent_slug": "discrepancy_detection"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
    }
    reqs = pending_requests(graph, payload={"s3_prefix": "docs/"}, answers={})
    assert reqs == []


def test_only_upstream_counts_not_downstream():
    """A node cannot be satisfied by something produced AFTER it."""
    graph = {
        "nodes": [
            {"id": "n1", "agent_slug": "discrepancy_detection"},
            {"id": "n2", "agent_slug": "data_extraction"},
        ],
        "edges": [{"source": "n1", "target": "n2"}],
    }
    reqs = pending_requests(graph, payload={}, answers={})
    assert any(r.node_id == "n1" for r in reqs)   # details not yet produced


def test_an_agent_with_no_elicit_contract_asks_for_nothing():
    graph = {"nodes": [{"id": "n1", "agent_slug": "opportunity_miner"}], "edges": []}
    assert pending_requests(graph, payload={}, answers={}) == []
