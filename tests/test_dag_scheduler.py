# tests/test_dag_scheduler.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


def _build_simple_graph():
    """Build: extract -> rank -> draft (linear DAG)."""
    from orchestration.workflow_engine import WorkflowGraph, WorkflowNode

    g = WorkflowGraph(name="test_linear")
    g.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    g.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
    g.add_node(WorkflowNode(name="draft", agent_type="email_drafting"))
    g.add_edge("extract", "rank")
    g.add_edge("rank", "draft")
    return g


def _build_parallel_graph():
    """Build: extract -> [eval, compare] -> draft (diamond DAG)."""
    from orchestration.workflow_engine import WorkflowGraph, WorkflowNode

    g = WorkflowGraph(name="test_parallel")
    g.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    g.add_node(WorkflowNode(name="eval", agent_type="quote_evaluation"))
    g.add_node(WorkflowNode(name="compare", agent_type="quote_comparison"))
    g.add_node(WorkflowNode(name="draft", agent_type="email_drafting"))
    g.add_edge("extract", "eval")
    g.add_edge("extract", "compare")
    g.add_edge("eval", "draft")
    g.add_edge("compare", "draft")
    return g


def test_initial_ready_nodes_linear():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_simple_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "pending", "rank": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["extract"]


def test_initial_ready_nodes_parallel():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "pending", "eval": "pending", "compare": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["extract"]


def test_parallel_dispatch_after_predecessor():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "pending", "compare": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert set(ready) == {"eval", "compare"}


def test_join_node_waits_for_all_predecessors():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "completed", "compare": "pending", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    # draft should NOT be ready because compare is still pending
    assert "draft" not in ready
    assert ready == ["compare"]


def test_join_node_ready_when_all_predecessors_done():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "completed", "compare": "completed", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["draft"]


def test_skipped_predecessor_unblocks_successor():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_parallel_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "eval": "completed", "compare": "skipped", "draft": "pending"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == ["draft"]


def test_workflow_complete_detection():
    from orchestration.dag_scheduler import DAGScheduler

    graph = _build_simple_graph()
    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "rank": "completed", "draft": "completed"}
    ready = scheduler.compute_ready_nodes(graph, node_statuses)
    assert ready == []
    assert scheduler.is_workflow_complete(graph, node_statuses)


def test_conditional_edge_skips_node():
    from orchestration.dag_scheduler import DAGScheduler
    from orchestration.workflow_engine import WorkflowGraph, WorkflowNode

    g = WorkflowGraph(name="test_conditional")
    g.add_node(WorkflowNode(name="extract", agent_type="data_extraction"))
    g.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
    g.add_edge("extract", "rank", condition=lambda state: False)  # Always skip

    scheduler = DAGScheduler(
        state_manager=MagicMock(),
        task_dispatcher=MagicMock(),
        agent_registry=MagicMock(),
    )
    node_statuses = {"extract": "completed", "rank": "pending"}
    shared_data = {}
    ready, skipped = scheduler.compute_ready_nodes_with_conditions(
        graph=g,
        node_statuses=node_statuses,
        shared_data=shared_data,
    )
    assert ready == []
    assert skipped == ["rank"]
