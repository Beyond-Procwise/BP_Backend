# tests/test_integration_workflow.py
"""Integration test: full workflow through DAG Scheduler with mocked I/O.

Validates the complete flow: start_workflow -> dispatch -> execute -> collect -> complete
without requiring Redis or PostgreSQL.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from orchestration.workflow_engine import WorkflowGraph, WorkflowNode
from orchestration.message_protocol import TaskMessage, ResultMessage
from orchestration.task_dispatcher import TaskDispatcher
from orchestration.dag_scheduler import DAGScheduler
from orchestration.result_collector import ResultCollector


class InMemoryStateManager:
    """StateManager substitute backed by dicts instead of PostgreSQL."""

    def __init__(self):
        self._executions = {}
        self._nodes = {}
        self._events = []
        self._next_id = 1

    def create_workflow_execution(self, workflow_id, workflow_name, user_id=None):
        eid = self._next_id
        self._next_id += 1
        self._executions[eid] = {
            "execution_id": eid, "workflow_id": workflow_id,
            "workflow_name": workflow_name, "user_id": user_id,
            "status": "pending", "shared_data": {}, "current_round": 0,
        }
        return eid

    def update_workflow_status(self, execution_id, status, completed_at=None):
        self._executions[execution_id]["status"] = status

    def get_workflow_execution(self, execution_id):
        return self._executions.get(execution_id)

    def get_active_workflows(self):
        return [e for e in self._executions.values() if e["status"] in ("pending", "running", "paused")]

    def increment_round(self, execution_id):
        self._executions[execution_id]["current_round"] += 1
        return self._executions[execution_id]["current_round"]

    def merge_shared_data(self, execution_id, new_fields):
        self._executions[execution_id]["shared_data"].update(new_fields)

    def create_node_executions(self, execution_id, nodes, round_num=0):
        for n in nodes:
            key = (execution_id, n["node_name"], round_num)
            self._nodes[key] = {"status": "pending", "agent_type": n["agent_type"], "attempt": 0}

    def update_node_status(self, execution_id, node_name, round_num, status, dispatched_at=None, attempt=None):
        key = (execution_id, node_name, round_num)
        if key in self._nodes:
            self._nodes[key]["status"] = status

    def record_node_result(self, execution_id, node_name, round_num, status, output_data=None, pass_fields=None, error=None, duration_ms=None):
        key = (execution_id, node_name, round_num)
        if key in self._nodes:
            self._nodes[key]["status"] = status
            self._nodes[key]["output_data"] = output_data
            self._nodes[key]["pass_fields"] = pass_fields

    def get_ready_nodes(self, execution_id, current_round):
        return [
            {"node_name": k[1], "agent_type": v["agent_type"], "attempt": v["attempt"]}
            for k, v in self._nodes.items()
            if k[0] == execution_id and k[2] == current_round and v["status"] == "ready"
        ]

    def get_node_statuses(self, execution_id, current_round):
        return {
            k[1]: v["status"]
            for k, v in self._nodes.items()
            if k[0] == execution_id and k[2] == current_round
        }

    def record_event(self, workflow_id, event_type, node_name=None, agent_type=None, payload=None, round_num=None):
        self._events.append({"workflow_id": workflow_id, "event_type": event_type, "node_name": node_name})

    def get_events(self, workflow_id, limit=100):
        return [e for e in self._events if e["workflow_id"] == workflow_id][:limit]


def test_linear_workflow_end_to_end():
    """extract -> rank -> draft: all succeed."""
    graph = WorkflowGraph(name="test_linear")
    graph.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    graph.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
    graph.add_node(WorkflowNode(name="draft", agent_type="email_drafting"))
    graph.add_edge("extract", "rank")
    graph.add_edge("rank", "draft")

    state_mgr = InMemoryStateManager()
    dispatched_tasks = []
    redis_mock = MagicMock()
    redis_mock.xadd = MagicMock()

    dispatcher = TaskDispatcher(redis_client=redis_mock, max_stream_len=1000)
    # Capture dispatched tasks
    original_dispatch = dispatcher.dispatch
    def capture_dispatch(task):
        dispatched_tasks.append(task)
        return original_dispatch(task)
    dispatcher.dispatch = capture_dispatch

    scheduler = DAGScheduler(
        state_manager=state_mgr,
        task_dispatcher=dispatcher,
        agent_registry=MagicMock(),
    )
    collector = ResultCollector(
        state_manager=state_mgr,
        dag_scheduler=scheduler,
        workflow_graphs={"test_linear": graph},
    )

    # Start workflow
    exec_id = scheduler.start_workflow(graph, {"query": "test"}, workflow_id="wf-e2e")
    assert state_mgr._executions[exec_id]["status"] == "running"
    assert len(dispatched_tasks) == 1
    assert dispatched_tasks[0].node_name == "extract"

    # Simulate extract completion
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[0].task_id, workflow_id="wf-e2e",
            node_name="extract", agent_type="data_extraction", status="SUCCESS",
            data={"details": "ok"}, pass_fields={"details": "ok"},
            next_agents=[], error=None, confidence=0.9,
            completed_at=datetime.now(timezone.utc), duration_ms=100,
        ),
        execution_id=exec_id,
    )
    assert len(dispatched_tasks) == 2
    assert dispatched_tasks[1].node_name == "rank"

    # Simulate rank completion
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[1].task_id, workflow_id="wf-e2e",
            node_name="rank", agent_type="supplier_ranking", status="SUCCESS",
            data={"ranking": [1]}, pass_fields={"ranking": [1]},
            next_agents=[], error=None, confidence=0.8,
            completed_at=datetime.now(timezone.utc), duration_ms=200,
        ),
        execution_id=exec_id,
    )
    assert len(dispatched_tasks) == 3
    assert dispatched_tasks[2].node_name == "draft"

    # Simulate draft completion
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[2].task_id, workflow_id="wf-e2e",
            node_name="draft", agent_type="email_drafting", status="SUCCESS",
            data={"drafts": ["email1"]}, pass_fields={},
            next_agents=[], error=None, confidence=0.95,
            completed_at=datetime.now(timezone.utc), duration_ms=300,
        ),
        execution_id=exec_id,
    )

    # Workflow should be complete
    assert state_mgr._executions[exec_id]["status"] == "completed"
    events = state_mgr.get_events("wf-e2e")
    event_types = [e["event_type"] for e in events]
    assert "workflow:started" in event_types
    assert "workflow:completed" in event_types


def test_parallel_dispatch():
    """extract -> [eval, compare]: both dispatched simultaneously."""
    graph = WorkflowGraph(name="test_parallel")
    graph.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    graph.add_node(WorkflowNode(name="eval", agent_type="quote_evaluation"))
    graph.add_node(WorkflowNode(name="compare", agent_type="quote_comparison"))
    graph.add_edge("extract", "eval")
    graph.add_edge("extract", "compare")

    state_mgr = InMemoryStateManager()
    dispatched_tasks = []
    redis_mock = MagicMock()

    dispatcher = TaskDispatcher(redis_client=redis_mock, max_stream_len=1000)
    original_dispatch = dispatcher.dispatch
    def capture_dispatch(task):
        dispatched_tasks.append(task)
        return original_dispatch(task)
    dispatcher.dispatch = capture_dispatch

    scheduler = DAGScheduler(
        state_manager=state_mgr,
        task_dispatcher=dispatcher,
        agent_registry=MagicMock(),
    )
    collector = ResultCollector(
        state_manager=state_mgr,
        dag_scheduler=scheduler,
        workflow_graphs={"test_parallel": graph},
    )

    exec_id = scheduler.start_workflow(graph, {}, workflow_id="wf-par")
    assert len(dispatched_tasks) == 1  # extract first

    # Complete extract
    collector.process_result(
        ResultMessage(
            task_id=dispatched_tasks[0].task_id, workflow_id="wf-par",
            node_name="extract", agent_type="data_extraction", status="SUCCESS",
            data={}, pass_fields={}, next_agents=[], error=None, confidence=0.9,
            completed_at=datetime.now(timezone.utc), duration_ms=100,
        ),
        execution_id=exec_id,
    )

    # Both eval and compare should now be dispatched
    assert len(dispatched_tasks) == 3
    dispatched_names = {t.node_name for t in dispatched_tasks[1:]}
    assert dispatched_names == {"eval", "compare"}
