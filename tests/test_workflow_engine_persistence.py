# tests/test_workflow_engine_persistence.py
"""P3: the declarative WorkflowEngine (the LIVE path behind
POST /agent-workflows/{id}/run) only ever wrote to a Redis checkpoint store.
proc.workflow_execution / proc.node_execution exist but stayed empty because
nothing on this path ever wrote to them -- StateManager writes them
correctly but is only reached via the (disabled) DAG scheduler.

These tests wire a mocked StateManager into WorkflowEngine and verify the
run trail is written: one proc.workflow_execution row per run, one
proc.node_execution row per node reflecting its real status, and that a
failed persistence write is logged rather than either crashing the
workflow or being silently swallowed.
"""
import logging

import pytest
from unittest.mock import MagicMock, call

from orchestration.workflow_engine import (
    WorkflowEngine,
    WorkflowGraph,
    WorkflowNode,
    WorkflowState,
    NodeStatus,
)
from agents.base_agent import AgentOutput, AgentStatus


def _make_engine(agents: dict, state_manager=None) -> WorkflowEngine:
    settings = MagicMock()
    settings.parallel_processing = False
    return WorkflowEngine(
        agent_registry=agents, settings=settings, state_manager=state_manager
    )


def _success_agent(data: dict) -> MagicMock:
    agent = MagicMock()
    agent.execute.return_value = AgentOutput(status=AgentStatus.SUCCESS, data=data)
    return agent


def _failing_agent(error: str) -> MagicMock:
    agent = MagicMock()
    agent.execute.return_value = AgentOutput(status=AgentStatus.FAILED, data={}, error=error)
    return agent


def _linear_graph() -> WorkflowGraph:
    g = WorkflowGraph(name="two_step")
    g.add_node(WorkflowNode(name="n1", agent_type="agent_a"))
    g.add_node(WorkflowNode(name="n2", agent_type="agent_b"))
    g.add_edge("n1", "n2")
    return g


def test_create_workflow_execution_called_once_per_run():
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    engine = _make_engine(
        {"agent_a": _success_agent({}), "agent_b": _success_agent({})},
        state_manager=state_manager,
    )
    state = engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    state_manager.create_workflow_execution.assert_called_once_with("wf-1", "two_step", "tester")
    assert state.execution_id == 42


def test_node_execution_row_created_for_each_node_before_it_runs():
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    engine = _make_engine(
        {"agent_a": _success_agent({}), "agent_b": _success_agent({})},
        state_manager=state_manager,
    )
    engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    created_nodes = [
        c.args[1][0]["node_name"] for c in state_manager.create_node_executions.call_args_list
    ]
    assert created_nodes == ["n1", "n2"]
    for c in state_manager.create_node_executions.call_args_list:
        assert c.args[0] == 42  # execution_id


def test_completed_node_result_recorded_with_status_completed():
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    engine = _make_engine(
        {"agent_a": _success_agent({"x": 1}), "agent_b": _success_agent({"y": 2})},
        state_manager=state_manager,
    )
    engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    recorded = {
        c.kwargs.get("node_name", c.args[1] if len(c.args) > 1 else None): c
        for c in state_manager.record_node_result.call_args_list
    }
    # record_node_result(execution_id, node_name, round_num, status, ...)
    calls_by_node = {c.args[1]: c for c in state_manager.record_node_result.call_args_list}
    assert calls_by_node["n1"].args[0] == 42
    assert calls_by_node["n1"].args[2] == 0  # round
    assert calls_by_node["n1"].args[3] == "completed"
    assert calls_by_node["n2"].args[3] == "completed"


def test_failed_node_result_recorded_with_error():
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    engine = _make_engine(
        {"agent_a": _failing_agent("boom"), "agent_b": _success_agent({})},
        state_manager=state_manager,
    )
    engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    calls_by_node = {c.args[1]: c for c in state_manager.record_node_result.call_args_list}
    assert calls_by_node["n1"].args[3] == "failed"
    assert calls_by_node["n1"].kwargs.get("error") == "boom"


def test_skipped_node_status_recorded():
    """A node whose predecessors are unmet (no traversable edges) is
    SKIPPED and must be recorded as such, not silently dropped."""
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    g = WorkflowGraph(name="skip_flow")
    g.add_node(WorkflowNode(name="mine", agent_type="miner", required=False))
    g.add_node(WorkflowNode(name="rank", agent_type="ranker"))
    g.add_edge("mine", "rank")

    # "miner" is absent from the registry -> mine is SKIPPED (agent missing).
    engine = _make_engine({"ranker": _success_agent({})}, state_manager=state_manager)
    state = engine.execute(g, input_data={}, user_id="tester", workflow_id="wf-1")

    assert state.node_statuses["mine"] == NodeStatus.SKIPPED
    calls_by_node = {c.args[1]: c for c in state_manager.record_node_result.call_args_list}
    assert calls_by_node["mine"].args[3] == "skipped"


def test_update_workflow_status_called_with_final_status():
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    engine = _make_engine(
        {"agent_a": _success_agent({}), "agent_b": _success_agent({})},
        state_manager=state_manager,
    )
    engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    state_manager.update_workflow_status.assert_called_once()
    args, kwargs = state_manager.update_workflow_status.call_args
    assert args[0] == 42
    assert args[1] == "completed"


def test_update_workflow_status_reports_failed_when_required_node_fails():
    state_manager = MagicMock()
    state_manager.create_workflow_execution.return_value = 42

    engine = _make_engine(
        {"agent_a": _failing_agent("boom"), "agent_b": _success_agent({})},
        state_manager=state_manager,
    )
    engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    args, kwargs = state_manager.update_workflow_status.call_args
    assert args[1] == "failed"


def test_resume_reuses_existing_execution_id_without_creating_a_new_row():
    state_manager = MagicMock()

    engine = _make_engine({"agent_a": _success_agent({}), "agent_b": _success_agent({})},
                           state_manager=state_manager)

    resume_state = WorkflowState(
        workflow_id="wf-1", workflow_name="two_step", user_id="tester",
    )
    resume_state.execution_id = 99
    resume_state.node_statuses["n1"] = NodeStatus.COMPLETED

    engine.execute(_linear_graph(), resume_state=resume_state)

    state_manager.create_workflow_execution.assert_not_called()
    args, kwargs = state_manager.update_workflow_status.call_args
    assert args[0] == 99


def test_db_write_failure_does_not_crash_the_workflow(caplog):
    """A failed audit write must be logged, not crash the run and not be
    silently swallowed."""
    state_manager = MagicMock()
    state_manager.create_workflow_execution.side_effect = RuntimeError("db is down")

    engine = _make_engine(
        {"agent_a": _success_agent({}), "agent_b": _success_agent({})},
        state_manager=state_manager,
    )

    with caplog.at_level(logging.WARNING):
        state = engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")

    # The workflow itself must still complete successfully.
    assert state.status == "completed"
    assert state.node_statuses["n1"] == NodeStatus.COMPLETED
    assert state.node_statuses["n2"] == NodeStatus.COMPLETED
    # But the failure must be visible in the logs, not silent.
    assert any("persist" in r.getMessage().lower() for r in caplog.records)


def test_no_state_manager_skips_persistence_without_error():
    """Default behaviour (state_manager=None) must be unchanged: no
    persistence attempted, no crash -- matches every pre-existing test in
    test_workflow_engine_bugs.py."""
    engine = _make_engine({"agent_a": _success_agent({}), "agent_b": _success_agent({})})
    state = engine.execute(_linear_graph(), input_data={}, user_id="tester", workflow_id="wf-1")
    assert state.status == "completed"
    assert state.execution_id is None
