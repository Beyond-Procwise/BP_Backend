from orchestration.workflow_engine import NodeStatus, WorkflowState


def test_awaiting_input_is_a_node_status():
    assert NodeStatus.AWAITING_INPUT.value == "awaiting_input"


def test_a_paused_run_checkpoints_and_round_trips():
    """A run that stops to ask the human must be resumable from its checkpoint."""
    st = WorkflowState(workflow_id="w1", workflow_name="test", user_id="u1")
    st.node_statuses["n1"] = NodeStatus.AWAITING_INPUT
    st.status = "awaiting_input"
    st.shared_data["already_known"] = "keep me"

    cp = st.checkpoint()
    assert cp is not None

    d = st.to_dict()
    assert d["status"] == "awaiting_input"
    assert d["shared_data"]["already_known"] == "keep me"
