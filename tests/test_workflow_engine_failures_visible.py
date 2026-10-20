"""Engine failures are visible (capability audit rec 6).

Before: a node that failed made every node after it "skipped" with an info-level log and no
reason; an optional node's failure left the run "completed" with nothing to say a step had
failed; nodes after a required failure had no record at all; and ``timeout_seconds`` was a
field nothing read.
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from agents.base_agent import AgentOutput, AgentStatus
from orchestration.workflow_engine import (
    NodeStatus, WorkflowEngine, WorkflowGraph, WorkflowNode,
)


def _engine(agents, state_manager=None):
    settings = MagicMock()
    settings.parallel_processing = False
    return WorkflowEngine(agent_registry=agents, settings=settings, state_manager=state_manager)


def _ok(data=None):
    a = MagicMock()
    a.execute.return_value = AgentOutput(status=AgentStatus.SUCCESS, data=data or {})
    return a


def _bad(error="boom"):
    a = MagicMock()
    a.execute.return_value = AgentOutput(status=AgentStatus.FAILED, data={}, error=error)
    return a


def _chain(*, a_required=True, **node_kw):
    g = WorkflowGraph(name="chain")
    g.add_node(WorkflowNode(name="A", agent_type="a", required=a_required, **node_kw))
    g.add_node(WorkflowNode(name="B", agent_type="b"))
    g.add_node(WorkflowNode(name="C", agent_type="c"))
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    return g


def _manager():
    m = MagicMock()
    m.create_workflow_execution.return_value = 7
    return m


def _recorded(m):
    """node_name -> (status, error) as written to proc.node_execution."""
    out = {}
    for c in m.record_node_result.call_args_list:
        _exec_id, name, _round, status = c.args[:4]
        out[name] = (status, c.kwargs.get("error"))
    return out


# --- an optional node fails: downstream is skipped WITH a reason, and the run says so -----

def test_downstream_of_a_failed_optional_node_is_skipped_with_the_reason():
    m = _manager()
    eng = _engine({"a": _bad("supplier API down"), "b": _ok(), "c": _ok()}, m)
    state = eng.execute(_chain(a_required=False), input_data={}, workflow_id="w")

    assert state.node_statuses["A"] == NodeStatus.FAILED
    assert state.node_statuses["B"] == NodeStatus.SKIPPED
    assert state.node_statuses["C"] == NodeStatus.SKIPPED
    assert "A" in state.skip_reasons["B"] and "failed" in state.skip_reasons["B"]
    # the cause is chained through descendants: C names the node that actually failed
    assert "A" in state.skip_reasons["C"]
    rec = _recorded(m)
    assert rec["B"][0] == "skipped" and "A" in rec["B"][1]
    assert rec["C"][0] == "skipped" and "A" in rec["C"][1]


def test_a_run_with_a_failed_node_says_so_even_when_the_node_was_optional():
    eng = _engine({"a": _bad(), "b": _ok(), "c": _ok()})
    state = eng.execute(_chain(a_required=False), input_data={}, workflow_id="w")
    assert state.status == "completed"            # the required path is unaffected
    assert state.failed_nodes == ["A"]
    assert state.degraded is True
    summary = state.to_dict()
    assert summary["failed_nodes"] == ["A"]
    assert summary["degraded"] is True
    assert set(summary["skipped_due_to_failure"]) == {"B", "C"}


def test_a_clean_run_is_not_degraded():
    eng = _engine({"a": _ok(), "b": _ok(), "c": _ok()})
    state = eng.execute(_chain(), input_data={}, workflow_id="w")
    assert state.degraded is False and state.failed_nodes == [] and state.skip_reasons == {}


def test_a_node_with_another_route_still_runs():
    g = WorkflowGraph(name="diamond")
    for n, t in (("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")):
        g.add_node(WorkflowNode(name=n, agent_type=t, required=(n != "A")))
    g.add_edge("A", "B")
    g.add_edge("C", "D")
    g.add_edge("B", "D")
    eng = _engine({"a": _bad(), "b": _ok(), "c": _ok(), "d": _ok()})
    state = eng.execute(g, input_data={}, workflow_id="w")
    assert state.node_statuses["B"] == NodeStatus.SKIPPED
    assert state.node_statuses["D"] == NodeStatus.COMPLETED   # reached through C


# --- a required node fails: the nodes that never ran are recorded, not absent -------------

def test_nodes_after_a_required_failure_are_recorded_as_skipped_with_why():
    m = _manager()
    eng = _engine({"a": _bad("no data"), "b": _ok(), "c": _ok()}, m)
    state = eng.execute(_chain(a_required=True), input_data={}, workflow_id="w")

    assert state.status == "failed"
    for name in ("B", "C"):
        assert state.node_statuses[name] == NodeStatus.SKIPPED
        assert "required node 'A' failed" in state.skip_reasons[name]
    rec = _recorded(m)
    assert rec["A"][0] == "failed"
    assert rec["B"][0] == "skipped" and "required node 'A' failed" in rec["B"][1]
    assert rec["C"][0] == "skipped"
    created = [c.args[1][0]["node_name"] for c in m.create_node_executions.call_args_list]
    assert created == ["A", "B", "C"]
    assert state.failed_nodes == ["A"]


def test_a_skip_that_is_not_caused_by_a_failure_has_no_failure_reason():
    g = WorkflowGraph(name="cond")
    g.add_node(WorkflowNode(name="A", agent_type="a"))
    g.add_node(WorkflowNode(name="B", agent_type="b"))
    g.add_edge("A", "B", condition=lambda s: False)
    state = _engine({"a": _ok(), "b": _ok()}).execute(g, input_data={}, workflow_id="w")
    assert state.node_statuses["B"] == NodeStatus.SKIPPED
    assert "B" not in state.skip_reasons
    assert state.degraded is False


# --- timeouts are enforced -------------------------------------------------------------

def _slow_agent(release: threading.Event, calls: list):
    a = MagicMock()

    def execute(ctx, **kw):
        calls.append(1)
        release.wait(5)
        return AgentOutput(status=AgentStatus.SUCCESS, data={"late": True})

    a.execute.side_effect = execute
    return a


def test_a_node_that_runs_past_its_timeout_fails_and_is_not_retried():
    release, calls = threading.Event(), []
    m = _manager()
    eng = _engine({"a": _slow_agent(release, calls), "b": _ok(), "c": _ok()}, m)
    started = time.monotonic()
    try:
        state = eng.execute(_chain(a_required=False, timeout_seconds=0.2, retry_count=2),
                            input_data={}, workflow_id="w")
    finally:
        release.set()
    assert time.monotonic() - started < 3
    assert state.node_statuses["A"] == NodeStatus.FAILED
    entry = next(e for e in state.errors if e["node"] == "A")
    assert "timed out after 0.2" in entry["error"] and entry["timed_out"] is True
    assert len(calls) == 1                          # a timed-out agent may still be running
    assert _recorded(m)["A"][0] == "timed_out"
    assert state.node_statuses["B"] == NodeStatus.SKIPPED
    assert "A" in state.skip_reasons["B"]


def test_a_node_inside_its_timeout_runs_normally_and_keeps_its_result():
    eng = _engine({"a": _ok({"x": 1}), "b": _ok(), "c": _ok()})
    state = eng.execute(_chain(timeout_seconds=5), input_data={}, workflow_id="w")
    assert state.node_statuses["A"] == NodeStatus.COMPLETED
    assert state.node_results["A"] == {"x": 1}


def test_an_exception_inside_a_timed_node_is_a_failure_not_a_hang():
    boom = MagicMock()
    boom.execute.side_effect = RuntimeError("kaput")
    state = _engine({"a": boom, "b": _ok(), "c": _ok()}).execute(
        _chain(a_required=False, timeout_seconds=5), input_data={}, workflow_id="w")
    entry = next(e for e in state.errors if e["node"] == "A")
    assert "kaput" in entry["error"] and not entry.get("timed_out")


def test_no_timeout_set_means_no_thread_and_no_limit():
    seen = {}

    def execute(ctx, **kw):
        seen["thread"] = threading.current_thread()
        return AgentOutput(status=AgentStatus.SUCCESS, data={})

    a = MagicMock()
    a.execute.side_effect = execute
    _engine({"a": a, "b": _ok(), "c": _ok()}).execute(_chain(), input_data={}, workflow_id="w")
    assert seen["thread"] is threading.current_thread()


def test_failed_state_round_trips_through_a_checkpoint():
    from orchestration.workflow_engine import WorkflowState
    state = _engine({"a": _bad(), "b": _ok(), "c": _ok()}).execute(
        _chain(a_required=False), input_data={}, workflow_id="w")
    back = WorkflowState.from_dict(state.to_dict())
    assert back.skip_reasons == state.skip_reasons
    assert back.failed_nodes == ["A"]
