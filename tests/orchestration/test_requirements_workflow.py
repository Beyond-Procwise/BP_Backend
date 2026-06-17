from orchestration.workflow_definitions import (
    build_requirements_to_ranking_workflow,
    get_workflow,
    list_workflows,
    _requirement_is_complete,
)
from orchestration.workflow_engine import WorkflowState


def _state(shared=None, results=None):
    s = WorkflowState(workflow_id="w1", workflow_name="requirements_to_ranking", user_id="u")
    s.shared_data = shared or {}
    s.node_results = results or {}
    return s


def test_builder_structure():
    g = build_requirements_to_ranking_workflow()
    assert g.name == "requirements_to_ranking"
    assert g.entry_node == "gather_requirement"
    assert g.nodes["gather_requirement"].agent_type == "requirements"
    assert g.nodes["rank_suppliers"].agent_type == "supplier_ranking"
    assert g.nodes["draft_emails"].agent_type == "email_drafting"
    # gather node surfaces the fields the downstream nodes + gate rely on
    shared = g.nodes["gather_requirement"].output_to_shared
    for f in ("complete", "query", "requirement"):
        assert f in shared
    # supplier_ranking pulls the query the agent emits on completion
    assert g.nodes["rank_suppliers"].input_mapping.get("gather_requirement.query") == "query"


def test_gate_only_traverses_when_complete():
    g = build_requirements_to_ranking_workflow()
    edge = next(e for e in g.edges if e.source == "gather_requirement" and e.target == "rank_suppliers")
    assert edge.condition is _requirement_is_complete
    # complete=True (from shared_data or node_results) → traverse
    assert _requirement_is_complete(_state(shared={"complete": True})) is True
    assert _requirement_is_complete(
        _state(results={"gather_requirement": {"complete": True}})) is True
    # still gathering → do not traverse to ranking
    assert _requirement_is_complete(_state(shared={"complete": False})) is False
    assert _requirement_is_complete(_state()) is False


def test_registered_in_registry():
    assert "requirements_to_ranking" in list_workflows()
    g = get_workflow("requirements_to_ranking")
    assert g.name == "requirements_to_ranking"
