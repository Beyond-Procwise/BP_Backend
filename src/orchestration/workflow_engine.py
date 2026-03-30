"""Declarative workflow engine following 12-Factor Agent principles.

This module replaces the monolithic orchestrator's hardcoded if/elif workflow
routing with a graph-based, declarative workflow engine.  Each workflow is
defined as a directed acyclic graph (DAG) of agent nodes with explicit
edges, conditions, and data mappings.

12-Factor Principles Applied:
  #5  Unify execution state and business state - single WorkflowState
  #6  Launch/Pause/Resume - checkpoints via Redis/serializable state
  #8  Own your control flow - explicit graph, no opaque loops
  #10 Small, focused agents - each node is a single agent
  #12 Stateless reducer - each step: (state, event) -> new_state
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from agents.base_agent import AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow State (12-Factor #5: unified execution + business state)
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowState:
    """Immutable-ish state that flows through the workflow graph.

    Following 12-Factor #12 (stateless reducer), each node receives the
    current state and produces a new state.  The state captures both
    execution metadata and business data.
    """

    workflow_id: str
    workflow_name: str
    user_id: str
    status: str = "running"
    current_node: Optional[str] = None
    node_statuses: Dict[str, NodeStatus] = field(default_factory=dict)
    node_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    routing_history: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    policy_context: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    task_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "user_id": self.user_id,
            "status": self.status,
            "current_node": self.current_node,
            "node_statuses": {k: v.value for k, v in self.node_statuses.items()},
            "node_results": self.node_results,
            "shared_data": self.shared_data,
            "routing_history": self.routing_history,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        state = cls(
            workflow_id=data["workflow_id"],
            workflow_name=data["workflow_name"],
            user_id=data.get("user_id", "system"),
        )
        state.status = data.get("status", "running")
        state.current_node = data.get("current_node")
        state.node_statuses = {
            k: NodeStatus(v) for k, v in data.get("node_statuses", {}).items()
        }
        state.node_results = data.get("node_results", {})
        state.shared_data = data.get("shared_data", {})
        state.routing_history = data.get("routing_history", [])
        state.errors = data.get("errors", [])
        state.started_at = data.get("started_at")
        state.completed_at = data.get("completed_at")
        return state

    def checkpoint(self) -> Dict[str, Any]:
        """Create a serializable checkpoint for pause/resume (#6)."""
        cp = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": self.to_dict(),
        }
        self.checkpoints.append(cp)
        return cp


# ---------------------------------------------------------------------------
# Workflow Graph Nodes and Edges
# ---------------------------------------------------------------------------

@dataclass
class WorkflowEdge:
    """A directed edge between two nodes in the workflow graph.

    Edges can have conditions (callables that receive the workflow state
    and return bool) to enable conditional branching.  Data mappings
    specify how output fields from the source node map to input fields
    of the target node.
    """

    source: str
    target: str
    condition: Optional[Callable[[WorkflowState], bool]] = None
    data_mapping: Dict[str, str] = field(default_factory=dict)
    label: str = ""

    def should_traverse(self, state: WorkflowState) -> bool:
        if self.condition is None:
            return True
        try:
            return self.condition(state)
        except Exception as exc:
            logger.warning(
                "Edge condition %s->%s failed: %s", self.source, self.target, exc
            )
            return False

    def map_data(self, state: WorkflowState) -> Dict[str, Any]:
        """Apply data mapping from source node output to target node input."""
        source_result = state.node_results.get(self.source, {})
        mapped: Dict[str, Any] = {}
        for source_field, target_field in self.data_mapping.items():
            value = source_result.get(source_field)
            if value is not None:
                mapped[target_field] = value
        return mapped


@dataclass
class WorkflowNode:
    """A node in the workflow graph representing a single agent execution.

    Each node wraps an agent type and defines how to build the agent's
    context from the workflow state.  Input mappings specify which fields
    from ``shared_data`` or previous node results flow into the agent.
    """

    name: str
    agent_type: str
    input_mapping: Dict[str, str] = field(default_factory=dict)
    static_inputs: Dict[str, Any] = field(default_factory=dict)
    output_to_shared: List[str] = field(default_factory=list)
    required: bool = True
    retry_count: int = 0
    timeout_seconds: Optional[int] = None

    def build_input_data(self, state: WorkflowState) -> Dict[str, Any]:
        """Build the agent's input_data from the workflow state."""
        input_data: Dict[str, Any] = dict(state.shared_data)
        input_data.update(self.static_inputs)

        # Apply explicit input mappings
        for source_path, target_field in self.input_mapping.items():
            parts = source_path.split(".")
            value: Any = state.node_results
            try:
                for part in parts:
                    if isinstance(value, dict):
                        value = value[part]
                    else:
                        value = None
                        break
            except (KeyError, TypeError):
                value = None
            if value is not None:
                input_data[target_field] = value

        return input_data


@dataclass
class WorkflowGraph:
    """Declarative workflow definition as a directed graph.

    Workflows are built by adding nodes and edges.  The engine validates
    the graph structure before execution (no cycles, all edge targets
    exist, etc.).
    """

    name: str
    description: str = ""
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: List[WorkflowEdge] = field(default_factory=list)
    entry_node: Optional[str] = None

    def add_node(self, node: WorkflowNode) -> "WorkflowGraph":
        self.nodes[node.name] = node
        if self.entry_node is None:
            self.entry_node = node.name
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        condition: Optional[Callable[[WorkflowState], bool]] = None,
        data_mapping: Optional[Dict[str, str]] = None,
        label: str = "",
    ) -> "WorkflowGraph":
        self.edges.append(
            WorkflowEdge(
                source=source,
                target=target,
                condition=condition,
                data_mapping=data_mapping or {},
                label=label,
            )
        )
        return self

    def get_successors(self, node_name: str) -> List[WorkflowEdge]:
        return [e for e in self.edges if e.source == node_name]

    def get_predecessors(self, node_name: str) -> List[WorkflowEdge]:
        return [e for e in self.edges if e.target == node_name]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the graph structure before execution."""
        issues: List[str] = []

        if not self.entry_node:
            issues.append("No entry node defined")
        elif self.entry_node not in self.nodes:
            issues.append(f"Entry node '{self.entry_node}' not found in nodes")

        for edge in self.edges:
            if edge.source not in self.nodes:
                issues.append(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in self.nodes:
                issues.append(f"Edge target '{edge.target}' not found in nodes")

        # Check for cycles using DFS
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for edge in self.get_successors(node):
                if edge.target not in visited:
                    if _has_cycle(edge.target):
                        return True
                elif edge.target in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node_name in self.nodes:
            if node_name not in visited:
                if _has_cycle(node_name):
                    issues.append("Workflow graph contains a cycle")
                    break

        return (len(issues) == 0, issues)

    def topological_order(self) -> List[str]:
        """Return nodes in topological order for sequential execution."""
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        for edge in self.edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for edge in self.get_successors(node):
                if edge.target in in_degree:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)
        return order


# ---------------------------------------------------------------------------
# Workflow Engine (12-Factor #8: own your control flow)
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """Executes workflow graphs with explicit, debuggable control flow.

    The engine walks the graph node by node, executing agents and
    following edges based on conditions.  State is checkpointed after
    each node for pause/resume capability.

    Usage::

        engine = WorkflowEngine(agent_registry, settings)
        graph = build_extraction_workflow()
        state = engine.execute(graph, input_data={"s3_prefix": "invoices/"})
    """

    def __init__(
        self,
        agent_registry: Dict[str, Any],
        settings: Any,
        *,
        checkpoint_store: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        manifest_service: Optional[Any] = None,
    ) -> None:
        self.agents = agent_registry
        self.settings = settings
        self._checkpoint_store = checkpoint_store
        self._event_bus = event_bus
        self._manifest_service = manifest_service

    def execute(
        self,
        graph: WorkflowGraph,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        user_id: str = "system",
        workflow_id: Optional[str] = None,
        resume_state: Optional[WorkflowState] = None,
    ) -> WorkflowState:
        """Execute a workflow graph and return the final state."""

        # Validate graph
        valid, issues = graph.validate()
        if not valid:
            raise ValueError(f"Invalid workflow graph: {'; '.join(issues)}")

        # Initialize or resume state (#6: launch/pause/resume)
        if resume_state:
            state = resume_state
            state.status = "running"
        else:
            state = WorkflowState(
                workflow_id=workflow_id or str(uuid.uuid4()),
                workflow_name=graph.name,
                user_id=user_id,
                started_at=datetime.utcnow().isoformat(),
                shared_data=dict(input_data or {}),
            )

        # Enrich with manifest if available
        if self._manifest_service:
            try:
                manifest = self._manifest_service.build_manifest(graph.name)
                state.policy_context = manifest.get("policies", [])
                state.knowledge_base = manifest.get("knowledge", {})
                state.task_profile = manifest.get("task", {})
                state.shared_data.setdefault("agent_manifest", manifest)
                state.shared_data.setdefault(
                    "policy_context", state.policy_context
                )
            except Exception:
                logger.debug("Manifest enrichment failed", exc_info=True)

        logger.info(
            "Starting workflow '%s' [%s]", graph.name, state.workflow_id
        )

        # Execute nodes in topological order
        execution_order = graph.topological_order()

        for node_name in execution_order:
            node = graph.nodes[node_name]

            # Check if already completed (for resume scenarios)
            if state.node_statuses.get(node_name) == NodeStatus.COMPLETED:
                continue

            # Check if predecessors completed successfully
            predecessors = graph.get_predecessors(node_name)
            if predecessors:
                # Check edge conditions
                any_edge_traversable = False
                for edge in predecessors:
                    source_status = state.node_statuses.get(edge.source)
                    if source_status == NodeStatus.COMPLETED and edge.should_traverse(state):
                        any_edge_traversable = True
                        # Apply data mappings from this edge
                        mapped_data = edge.map_data(state)
                        state.shared_data.update(mapped_data)

                if not any_edge_traversable:
                    state.node_statuses[node_name] = NodeStatus.SKIPPED
                    logger.info("Skipping node '%s' - no traversable edges", node_name)
                    continue

            # Execute the node
            state = self._execute_node(graph, node, state)

            # Checkpoint after each node (#6)
            self._save_checkpoint(state)

            # Stop on critical failure
            if (
                state.node_statuses.get(node_name) == NodeStatus.FAILED
                and node.required
            ):
                state.status = "failed"
                logger.error(
                    "Workflow '%s' failed at node '%s'",
                    graph.name,
                    node_name,
                )
                break

        # Finalize
        if state.status == "running":
            state.status = "completed"
        state.completed_at = datetime.utcnow().isoformat()

        logger.info(
            "Workflow '%s' [%s] finished with status: %s",
            graph.name,
            state.workflow_id,
            state.status,
        )

        return state

    def _execute_node(
        self,
        graph: WorkflowGraph,
        node: WorkflowNode,
        state: WorkflowState,
    ) -> WorkflowState:
        """Execute a single workflow node (12-Factor #12: stateless reducer)."""

        state.current_node = node.name
        state.node_statuses[node.name] = NodeStatus.RUNNING
        state.routing_history.append(node.name)

        agent = self.agents.get(node.agent_type)
        if agent is None:
            if node.required:
                state.node_statuses[node.name] = NodeStatus.FAILED
                state.errors.append({
                    "node": node.name,
                    "error": f"Agent '{node.agent_type}' not found in registry",
                })
            else:
                state.node_statuses[node.name] = NodeStatus.SKIPPED
            return state

        # Build agent context from workflow state
        input_data = node.build_input_data(state)
        context = AgentContext(
            workflow_id=state.workflow_id,
            agent_id=node.agent_type,
            user_id=state.user_id,
            input_data=input_data,
            parent_agent=state.routing_history[-2] if len(state.routing_history) > 1 else None,
            routing_history=list(state.routing_history),
            task_profile=dict(state.task_profile),
            policy_context=list(state.policy_context),
            knowledge_base=dict(state.knowledge_base),
        )

        logger.info("Executing node '%s' (agent: %s)", node.name, node.agent_type)

        # Execute with retry support
        result: Optional[AgentOutput] = None
        attempts = node.retry_count + 1
        for attempt in range(attempts):
            try:
                result = agent.execute(context)
                if result.status == AgentStatus.SUCCESS:
                    break
                if attempt < attempts - 1:
                    logger.warning(
                        "Node '%s' attempt %d failed, retrying...",
                        node.name,
                        attempt + 1,
                    )
            except Exception as exc:
                logger.exception("Node '%s' execution error", node.name)
                result = AgentOutput(
                    status=AgentStatus.FAILED,
                    data={},
                    error=str(exc),
                )

        if result is None:
            result = AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error="No result produced",
            )

        # Update state with results
        state.node_results[node.name] = result.data or {}
        if result.status == AgentStatus.SUCCESS:
            state.node_statuses[node.name] = NodeStatus.COMPLETED
            # Copy specified output fields to shared data
            for output_field in node.output_to_shared:
                value = result.data.get(output_field)
                if value is not None:
                    state.shared_data[output_field] = value
            # Also merge pass_fields
            if result.pass_fields:
                state.shared_data.update(result.pass_fields)
        else:
            state.node_statuses[node.name] = NodeStatus.FAILED
            state.errors.append({
                "node": node.name,
                "agent": node.agent_type,
                "error": result.error or "Unknown error",
                "data": result.data,
            })

        return state

    def _save_checkpoint(self, state: WorkflowState) -> None:
        """Persist a checkpoint for pause/resume (#6)."""
        if not self._checkpoint_store:
            return
        try:
            checkpoint = state.checkpoint()
            self._checkpoint_store.set(
                f"workflow_checkpoint:{state.workflow_id}",
                json.dumps(checkpoint, default=str),
            )
        except Exception:
            logger.debug("Checkpoint save failed", exc_info=True)

    def resume(
        self,
        workflow_id: str,
        graph: WorkflowGraph,
    ) -> Optional[WorkflowState]:
        """Resume a paused workflow from its last checkpoint (#6)."""
        if not self._checkpoint_store:
            return None
        try:
            raw = self._checkpoint_store.get(f"workflow_checkpoint:{workflow_id}")
            if not raw:
                return None
            checkpoint = json.loads(raw)
            state = WorkflowState.from_dict(checkpoint["state"])
            return self.execute(graph, resume_state=state)
        except Exception:
            logger.exception("Failed to resume workflow %s", workflow_id)
            return None
