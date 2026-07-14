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
    """Real member of the group (s3_object_keys, the exact-keys list the document
    picker submits) satisfies it. This USED to assert the opposite of real
    behaviour: it answered with document_ids, a field that was in the any_of
    but which the agent never read and could never actually be answered
    (pending_requests always asks for any_of[0]) — so the old assertion
    certified a guarantee the code could not honour."""
    reqs = pending_requests(
        EXTRACT_THEN_RANK, payload={"s3_object_keys": ["documents/workspace/a.pdf"]}, answers={}
    )
    assert not [r for r in reqs if r.node_id == "n1"]


def test_document_ids_no_longer_satisfies_the_group():
    """document_ids was removed from data_extraction's any_of (CRITICAL 2): it was
    unanswerable (any_of[0] is always the required_field asked for) and unread by
    the agent, so answering it used to silently satisfy the group and let a run
    sweep the entire default corpus. It must now still be asked for."""
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={"document_ids": [1, 2]}, answers={})
    assert [r.node_id for r in reqs if r.node_id == "n1"] == ["n1"]


def test_a_blank_value_does_not_satisfy_a_group():
    """CRITICAL 2, THE regression guard: a key's mere PRESENCE in the payload used
    to satisfy a group, even when its value was blank. A UI form field submitted
    empty (or a caller sending {"s3_prefix": ""}) must still be asked for."""
    reqs = pending_requests(
        EXTRACT_THEN_RANK,
        payload={"s3_prefix": "", "s3_object_key": None, "s3_object_keys": []},
        answers={},
    )
    assert [r.node_id for r in reqs if r.node_id == "n1"] == ["n1"]


def test_a_blank_answer_does_not_satisfy_a_group():
    reqs = pending_requests(EXTRACT_THEN_RANK, payload={}, answers={"s3_prefix": ""})
    assert [r.node_id for r in reqs if r.node_id == "n1"] == ["n1"]


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
