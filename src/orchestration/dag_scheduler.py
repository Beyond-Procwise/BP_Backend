# src/orchestration/dag_scheduler.py
"""DAG-based workflow scheduler with parallel node dispatch.

Replaces the sequential WorkflowEngine executor. Uses WorkflowGraph
data structures from workflow_engine.py but executes nodes in parallel
via TaskDispatcher + Redis Streams.

Spec reference: Sections 1, 3 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orchestration.message_protocol import EventMessage, TaskMessage
from orchestration.state_manager import StateManager
from orchestration.task_dispatcher import TaskDispatcher
from orchestration.workflow_engine import NodeStatus, WorkflowEdge, WorkflowGraph, WorkflowState

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({"completed", "skipped", "failed", "timed_out"})
_DONE_STATUSES = frozenset({"completed", "skipped"})


class DAGScheduler:
    def __init__(
        self,
        state_manager: StateManager,
        task_dispatcher: TaskDispatcher,
        agent_registry: Any,
        manifest_service: Any = None,
        default_timeout: int = 300,
    ):
        self._state = state_manager
        self._dispatcher = task_dispatcher
        self._agents = agent_registry
        self._manifest = manifest_service
        self._default_timeout = default_timeout

    # ── Public API ──────────────────────────────────────────────

    def start_workflow(
        self,
        graph: WorkflowGraph,
        input_data: Dict[str, Any],
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        workflow_id = workflow_id or f"wf-{uuid.uuid4().hex[:12]}"
        execution_id = self._state.create_workflow_execution(
            workflow_id=workflow_id,
            workflow_name=graph.name,
            user_id=user_id,
        )
        self._state.merge_shared_data(execution_id, input_data)
        nodes = [
            {"node_name": name, "agent_type": node.agent_type}
            for name, node in graph.nodes.items()
        ]
        self._state.create_node_executions(execution_id, nodes, round_num=0)
        self._state.update_workflow_status(execution_id, "running")
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:started",
            payload={"workflow_name": graph.name, "node_count": len(nodes)},
        )
        self._evaluate_and_dispatch(graph, execution_id, workflow_id, round_num=0)
        return execution_id

    def on_node_completed(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        node_name: str,
        round_num: int,
    ) -> None:
        node_statuses = self._state.get_node_statuses(execution_id, round_num)
        wf = self._state.get_workflow_execution(execution_id)
        shared_data = wf["shared_data"] if wf else {}

        ready, skipped = self.compute_ready_nodes_with_conditions(
            graph, node_statuses, shared_data,
        )
        for skip_name in skipped:
            self._state.update_node_status(execution_id, skip_name, round_num, "skipped")
            self._state.record_event(
                workflow_id=workflow_id,
                event_type="node:skipped",
                node_name=skip_name,
                round_num=round_num,
            )

        if ready:
            self._dispatch_nodes(graph, execution_id, workflow_id, ready, round_num, shared_data)
        elif self.is_workflow_complete(graph, node_statuses):
            self._complete_workflow(execution_id, workflow_id)
        elif self._has_failed_required(graph, node_statuses):
            self._fail_workflow(execution_id, workflow_id, graph, node_statuses)

    def on_node_failed(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        node_name: str,
        round_num: int,
    ) -> None:
        node = graph.nodes.get(node_name)
        if node and node.required:
            self._fail_workflow(execution_id, workflow_id, graph,
                                self._state.get_node_statuses(execution_id, round_num))
        else:
            self._state.update_node_status(execution_id, node_name, round_num, "skipped")
            self.on_node_completed(graph, execution_id, workflow_id, node_name, round_num)

    # ── Negotiation Re-entry ────────────────────────────────────

    def start_next_round(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        subgraph_nodes: List[str],
    ) -> int:
        new_round = self._state.increment_round(execution_id)
        nodes = [
            {"node_name": name, "agent_type": graph.nodes[name].agent_type}
            for name in subgraph_nodes
            if name in graph.nodes
        ]
        self._state.create_node_executions(execution_id, nodes, round_num=new_round)
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:resumed",
            payload={"round": new_round, "nodes": subgraph_nodes},
            round_num=new_round,
        )
        self._evaluate_and_dispatch(graph, execution_id, workflow_id, round_num=new_round)
        return new_round

    # ── Ready-Node Computation ──────────────────────────────────

    def compute_ready_nodes(
        self, graph: WorkflowGraph, node_statuses: Dict[str, str]
    ) -> List[str]:
        ready, _ = self.compute_ready_nodes_with_conditions(graph, node_statuses, {})
        return ready

    def compute_ready_nodes_with_conditions(
        self,
        graph: WorkflowGraph,
        node_statuses: Dict[str, str],
        shared_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        predecessors = self._build_predecessor_map(graph)
        incoming_edges = self._build_incoming_edge_map(graph)
        ready = []
        skipped = []

        for node_name, status in node_statuses.items():
            if status != "pending":
                continue
            preds = predecessors.get(node_name, set())
            if not preds:
                ready.append(node_name)
                continue
            all_preds_done = all(
                node_statuses.get(p) in _DONE_STATUSES for p in preds
            )
            if not all_preds_done:
                continue
            edges_to_this = incoming_edges.get(node_name, [])
            if edges_to_this and shared_data is not None:
                # Convert string statuses to NodeStatus enums for WorkflowState
                _STATUS_MAP = {
                    "pending": NodeStatus.PENDING, "ready": NodeStatus.PENDING,
                    "running": NodeStatus.RUNNING, "completed": NodeStatus.COMPLETED,
                    "failed": NodeStatus.FAILED, "skipped": NodeStatus.SKIPPED,
                    "timed_out": NodeStatus.FAILED,
                }
                enum_statuses = {
                    k: _STATUS_MAP.get(v, NodeStatus.PENDING)
                    for k, v in node_statuses.items()
                }
                mock_state = WorkflowState(
                    workflow_id="", workflow_name="", user_id="",
                    status="running", current_node=None,
                    node_statuses=enum_statuses,
                    node_results={}, shared_data=shared_data or {},
                    routing_history=[], errors=[], checkpoints=[],
                )
                any_traversable = any(
                    e.should_traverse(mock_state) for e in edges_to_this
                    if e.condition is not None
                )
                has_conditions = any(e.condition is not None for e in edges_to_this)
                if has_conditions and not any_traversable:
                    skipped.append(node_name)
                    continue
            ready.append(node_name)

        return ready, skipped

    def is_workflow_complete(
        self, graph: WorkflowGraph, node_statuses: Dict[str, str]
    ) -> bool:
        return all(
            status in _TERMINAL_STATUSES
            for status in node_statuses.values()
        )

    # ── Internal Helpers ────────────────────────────────────────

    def _evaluate_and_dispatch(
        self, graph: WorkflowGraph, execution_id: int, workflow_id: str, round_num: int
    ) -> None:
        node_statuses = self._state.get_node_statuses(execution_id, round_num)
        wf = self._state.get_workflow_execution(execution_id)
        shared_data = wf["shared_data"] if wf else {}
        ready, skipped = self.compute_ready_nodes_with_conditions(
            graph, node_statuses, shared_data,
        )
        for skip_name in skipped:
            self._state.update_node_status(execution_id, skip_name, round_num, "skipped")
        if ready:
            self._dispatch_nodes(graph, execution_id, workflow_id, ready, round_num, shared_data)

    def _dispatch_nodes(
        self,
        graph: WorkflowGraph,
        execution_id: int,
        workflow_id: str,
        node_names: List[str],
        round_num: int,
        shared_data: Dict[str, Any],
    ) -> None:
        for node_name in node_names:
            node = graph.nodes[node_name]
            task_id = f"t-{uuid.uuid4().hex[:12]}"
            context = {
                "workflow_id": workflow_id,
                "agent_id": node.agent_type,
                "user_id": "",
                "input_data": node.build_input_data(
                    WorkflowState(
                        workflow_id=workflow_id, workflow_name=graph.name,
                        user_id="", status="running", current_node=node_name,
                        node_statuses={}, node_results={},
                        shared_data=shared_data, routing_history=[],
                        errors=[], checkpoints=[],
                    )
                ),
                "policy_context": [],
                "knowledge_base": {},
                "routing_history": [],
                "task_profile": {},
                "task_id": task_id,
            }
            msg = TaskMessage(
                task_id=task_id,
                workflow_id=workflow_id,
                node_name=node_name,
                agent_type=node.agent_type,
                context=context,
                priority="normal",
                dispatched_at=datetime.now(timezone.utc),
                timeout_seconds=node.timeout_seconds or self._default_timeout,
                attempt=1,
            )
            self._state.update_node_status(
                execution_id, node_name, round_num, "running",
                dispatched_at=msg.dispatched_at, attempt=1,
            )
            self._dispatcher.dispatch(msg)

    def _complete_workflow(self, execution_id: int, workflow_id: str) -> None:
        now = datetime.now(timezone.utc)
        self._state.update_workflow_status(execution_id, "completed", completed_at=now)
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:completed",
            payload={},
        )
        logger.info("Workflow %s completed (execution_id=%d)", workflow_id, execution_id)

    def _fail_workflow(
        self, execution_id: int, workflow_id: str,
        graph: WorkflowGraph, node_statuses: Dict[str, str],
    ) -> None:
        now = datetime.now(timezone.utc)
        self._state.update_workflow_status(execution_id, "failed", completed_at=now)
        running_types = [
            graph.nodes[n].agent_type
            for n, s in node_statuses.items()
            if s == "running"
        ]
        if running_types:
            self._dispatcher.publish_cancellation(workflow_id, running_types)
        self._state.record_event(
            workflow_id=workflow_id,
            event_type="workflow:failed",
            payload={"reason": "required_node_failed"},
        )
        logger.error("Workflow %s failed (execution_id=%d)", workflow_id, execution_id)

    def _has_failed_required(
        self, graph: WorkflowGraph, node_statuses: Dict[str, str]
    ) -> bool:
        for name, status in node_statuses.items():
            if status in ("failed", "timed_out"):
                node = graph.nodes.get(name)
                if node and node.required:
                    return True
        return False

    @staticmethod
    def _build_predecessor_map(graph: WorkflowGraph) -> Dict[str, set]:
        preds: Dict[str, set] = {name: set() for name in graph.nodes}
        for edge in graph.edges:
            preds[edge.target].add(edge.source)
        return preds

    @staticmethod
    def _build_incoming_edge_map(graph: WorkflowGraph) -> Dict[str, List[WorkflowEdge]]:
        incoming: Dict[str, List[WorkflowEdge]] = {name: [] for name in graph.nodes}
        for edge in graph.edges:
            incoming[edge.target].append(edge)
        return incoming
