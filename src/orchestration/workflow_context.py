"""Shared blackboard for inter-agent communication within a workflow.

WorkflowContext acts as the central shared state object passed through an
agentic pipeline. Any agent can read prior outputs, contribute new data,
and emit structured signals to influence downstream routing decisions.

Spec reference: Task 2 of the agentic re-engineering plan.
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(str, Enum):
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    CONFIDENCE_LOW = "CONFIDENCE_LOW"
    RECOMMEND_ESCALATION = "RECOMMEND_ESCALATION"
    SUGGEST_AGENT = "SUGGEST_AGENT"


@dataclass
class AgentSignal:
    """Structured signal emitted by an agent during workflow execution."""

    agent: str
    signal_type: SignalType
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "signal_type": self.signal_type.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class WorkflowContext:
    """Shared blackboard for all agents participating in a single workflow run.

    Agents read from this context to understand prior work, write their own
    results, and emit signals that affect downstream routing or escalation.

    Attributes:
        workflow_id: Unique identifier for this workflow run.
        goal: High-level natural language goal driving the workflow.
        escalation_policy: Policy config controlling automatic escalation.
        agent_results: Ordered mapping of agent_name → output dict (insertion order
            reflects execution order, giving each agent the full prior chain).
        shared_data: Accumulator for cross-agent facts that any agent may read or
            extend (e.g. extracted entities, resolved supplier IDs).
    """

    def __init__(
        self,
        goal: str,
        workflow_id: Optional[str] = None,
        escalation_policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.workflow_id: str = workflow_id or str(uuid.uuid4())
        self.goal: str = goal
        self.escalation_policy: Dict[str, Any] = escalation_policy or {}
        self.agent_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.shared_data: Dict[str, Any] = {}
        self._signals: List[AgentSignal] = []

    # ── Result Management ───────────────────────────────────────────────────

    def record_result(self, agent_name: str, output: Dict[str, Any]) -> None:
        """Store the output dict produced by *agent_name*.

        Subsequent agents can retrieve this via :meth:`get_prior_result`.
        Results are kept in insertion order so the full chain is preserved.

        Args:
            agent_name: Name of the agent that produced the result.
            output: Arbitrary output dictionary from the agent.
        """
        self.agent_results[agent_name] = output

    def get_prior_result(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Return the stored output for *agent_name*, or None if not present.

        Args:
            agent_name: Name of the agent whose result to retrieve.
        """
        return self.agent_results.get(agent_name)

    # ── Shared Data ─────────────────────────────────────────────────────────

    def update_shared(self, key: str, value: Any) -> None:
        """Set or overwrite a key in the shared data accumulator.

        Args:
            key: Shared data key.
            value: Value to store under *key*.
        """
        self.shared_data[key] = value

    # ── Procurement Brief ───────────────────────────────────────────────────

    def set_procurement_brief(self, brief: Dict[str, Any]) -> None:
        """Convenience method to store a structured procurement brief.

        Stores the brief under the reserved key ``"procurement_brief"`` in
        :attr:`shared_data`.

        Args:
            brief: Structured procurement brief dictionary.
        """
        self.shared_data["procurement_brief"] = brief

    def get_procurement_brief(self) -> Optional[Dict[str, Any]]:
        """Return the stored procurement brief, or None."""
        return self.shared_data.get("procurement_brief")

    # ── Signal Bus ──────────────────────────────────────────────────────────

    def emit_signal(
        self,
        agent: str,
        signal_type: SignalType,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentSignal:
        """Emit a typed signal from *agent* onto the workflow signal bus.

        Signals are visible to the orchestrator and to downstream agents via
        :meth:`get_signals` / :meth:`has_signal`.

        Args:
            agent: Name of the agent emitting the signal.
            signal_type: One of the :class:`SignalType` members.
            message: Human-readable description of the signal.
            data: Optional payload dictionary attached to the signal.

        Returns:
            The :class:`AgentSignal` instance that was recorded.
        """
        sig = AgentSignal(
            agent=agent,
            signal_type=signal_type,
            message=message,
            data=data or {},
        )
        self._signals.append(sig)
        return sig

    def get_signals(
        self,
        signal_type: Optional[SignalType] = None,
        agent: Optional[str] = None,
    ) -> List[AgentSignal]:
        """Return signals, optionally filtered by *signal_type* and/or *agent*.

        Filters are ANDed: supplying both ``signal_type`` and ``agent`` returns
        only signals that match both criteria.

        Args:
            signal_type: When provided, only signals of this type are returned.
            agent: When provided, only signals from this agent are returned.

        Returns:
            List of matching :class:`AgentSignal` instances in emission order.
        """
        results = self._signals
        if signal_type is not None:
            results = [s for s in results if s.signal_type == signal_type]
        if agent is not None:
            results = [s for s in results if s.agent == agent]
        return results

    def has_signal(
        self,
        signal_type: Optional[SignalType] = None,
        agent: Optional[str] = None,
    ) -> bool:
        """Return True if at least one signal matches the given filters.

        Args:
            signal_type: Optional signal type filter.
            agent: Optional agent name filter.
        """
        return len(self.get_signals(signal_type=signal_type, agent=agent)) > 0

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire context to a plain dictionary.

        The result is suitable for JSON serialisation and for passing as the
        full context payload to the next agent in the chain.

        Returns:
            Dictionary containing all workflow context state.
        """
        return {
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "escalation_policy": self.escalation_policy,
            "agent_results": dict(self.agent_results),
            "shared_data": self.shared_data,
            "signals": [s.to_dict() for s in self._signals],
        }
