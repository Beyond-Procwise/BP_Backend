"""A workflow whose engine run failed is reported as failed (capability audit rec 6).

``execute_workflow`` used to copy the engine's errors into the result and then carry on as though
the run had succeeded: it closed the trail, published the completion event, finalised the
workflow memory and fed the learning loop all as "completed", and returned status "completed".
"""
from types import SimpleNamespace

import pytest

from orchestration.orchestrator import Orchestrator
from orchestration.workflow_engine import NodeStatus, WorkflowState
from orchestration.workflow_definitions import WORKFLOW_REGISTRY
from tests.orchestration.test_supplier_interaction_needs_the_engine import (  # noqa: F401
    StubAgent, StubNick, patch_dependencies,
)

NAME = "supplier_interaction"


def _state(status="completed", **kw):
    st = WorkflowState(workflow_id="w", workflow_name=NAME, user_id="u")
    st.status = status
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _orch(state, monkeypatch):
    orch = Orchestrator(StubNick({NAME: StubAgent()}))
    orch._workflow_engine = SimpleNamespace(execute=lambda graph, **k: state)
    orch._workflow_registry = {NAME: WORKFLOW_REGISTRY[NAME]}
    seen = SimpleNamespace(published=[], memory=[], learned=[], trail=[])
    monkeypatch.setattr(orch, "_publish_workflow_complete",
                        lambda **kw: seen.published.append(kw["status"]))
    monkeypatch.setattr(orch, "_finalise_workflow_memory",
                        lambda **kw: seen.memory.append(kw["success"]))
    monkeypatch.setattr(orch, "_learn_from_workflow",
                        lambda *a, **kw: seen.learned.append(kw["success"]))
    monkeypatch.setattr(orch, "_close_run_trail", lambda tid, status: seen.trail.append(status))
    return orch, seen


def test_a_failed_engine_run_is_reported_failed_everywhere(monkeypatch):
    failed = _state("failed", errors=[{"node": "rank", "error": "no suppliers"}])
    failed.node_statuses = {"rank": NodeStatus.FAILED}
    orch, seen = _orch(failed, monkeypatch)

    out = orch.execute_workflow(NAME, {})

    assert out["status"] == "failed"
    assert "rank" in out["error"] and "no suppliers" in out["error"]
    assert out["errors"] == failed.errors
    assert "engine_state" in out["result"]            # the detail a caller needs is still there
    assert seen.published == ["failed"]
    assert seen.memory == [False]
    assert seen.learned == [False]
    assert seen.trail == ["failed"]


def test_a_completed_run_is_unchanged(monkeypatch):
    orch, seen = _orch(_state("completed"), monkeypatch)
    out = orch.execute_workflow(NAME, {})
    assert out["status"] == "completed"
    assert "degraded" not in out and "failed_nodes" not in out
    assert seen.published == ["completed"] and seen.memory == [True] and seen.learned == [True]


def test_a_run_that_carried_on_past_a_failed_optional_node_says_so(monkeypatch):
    st = _state("completed", errors=[{"node": "enrich", "error": "timed out"}])
    st.node_statuses = {"enrich": NodeStatus.FAILED, "rank": NodeStatus.COMPLETED}
    orch, seen = _orch(st, monkeypatch)

    out = orch.execute_workflow(NAME, {})

    assert out["status"] == "completed"               # the required path did finish
    assert out["degraded"] is True
    assert out["failed_nodes"] == ["enrich"]
    assert seen.published == ["completed"]
