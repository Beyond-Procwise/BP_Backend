"""Canonical agent interface and contracts for the ProcWise agentic framework.

Every agent in the system implements :class:`AgentInterface` which defines
the contract that the workflow engine relies on.  Agents declare their
capabilities, required inputs, and produced outputs so the orchestration
layer can validate wiring at graph-build time rather than failing at runtime.

This module also provides :class:`AgentCapability` descriptors and the
:class:`AgentContract` dataclass that captures the full specification of
what an agent can do and what it needs.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from agents.base_agent import AgentContext, AgentOutput, AgentStatus

logger = logging.getLogger(__name__)


class AgentCapability(str, Enum):
    """Declarative capabilities that agents expose to the workflow engine."""

    DOCUMENT_EXTRACTION = "document_extraction"
    DOCUMENT_CLASSIFICATION = "document_classification"
    HEADER_EXTRACTION = "header_extraction"
    LINE_ITEM_EXTRACTION = "line_item_extraction"
    SUPPLIER_RANKING = "supplier_ranking"
    QUOTE_EVALUATION = "quote_evaluation"
    QUOTE_COMPARISON = "quote_comparison"
    OPPORTUNITY_MINING = "opportunity_mining"
    EMAIL_DRAFTING = "email_drafting"
    EMAIL_DISPATCH = "email_dispatch"
    EMAIL_WATCHING = "email_watching"
    NEGOTIATION = "negotiation"
    SUPPLIER_INTERACTION = "supplier_interaction"
    DISCREPANCY_DETECTION = "discrepancy_detection"
    APPROVAL_DECISION = "approval_decision"
    RAG_QUERY = "rag_query"


@dataclass(frozen=True)
class AgentContract:
    """Specification of an agent's interface for the workflow engine.

    The contract declares what data the agent requires (``required_inputs``),
    what it optionally accepts (``optional_inputs``), what it produces
    (``output_fields``), and what capabilities it exposes.  The workflow
    engine uses this metadata to validate pipeline wiring before execution.
    """

    agent_type: str
    capabilities: FrozenSet[AgentCapability]
    required_inputs: FrozenSet[str] = frozenset()
    optional_inputs: FrozenSet[str] = frozenset()
    output_fields: FrozenSet[str] = frozenset()
    description: str = ""
    version: str = "1.0.0"

    def validate_inputs(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check whether ``input_data`` satisfies the required inputs."""
        missing = [
            field_name
            for field_name in self.required_inputs
            if field_name not in input_data or input_data[field_name] is None
        ]
        return (len(missing) == 0, missing)


class AgentInterface(abc.ABC):
    """Abstract interface that all ProcWise agents must implement.

    This provides a clean separation between the agent's business logic
    (``process``) and the framework concerns handled by ``execute``.
    Agents that implement this interface are automatically reusable across
    different workflow configurations.
    """

    @property
    @abc.abstractmethod
    def contract(self) -> AgentContract:
        """Return the agent's contract describing its capabilities and I/O."""
        ...

    @abc.abstractmethod
    def process(self, context: AgentContext) -> AgentOutput:
        """Execute the agent's core business logic.

        Unlike ``run``, this method is called by the workflow engine with
        validated inputs.  Implementations should focus purely on their
        domain logic without worrying about routing or logging.
        """
        ...

    @property
    def agent_type(self) -> str:
        return self.contract.agent_type

    @property
    def capabilities(self) -> FrozenSet[AgentCapability]:
        return self.contract.capabilities

    def can_handle(self, capability: AgentCapability) -> bool:
        return capability in self.contract.capabilities

    def validate_context(self, context: AgentContext) -> Tuple[bool, List[str]]:
        """Validate that the context has the required inputs for this agent."""
        return self.contract.validate_inputs(context.input_data)


# ---------------------------------------------------------------------------
# Agent role categories for workflow composition
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    """Logical roles agents play in a procurement workflow."""

    SOURCE = "source"            # Produces initial data (extraction, mining)
    PROCESSOR = "processor"      # Transforms/enriches data (ranking, evaluation)
    COMMUNICATOR = "communicator" # Handles external comms (email, dispatch)
    VALIDATOR = "validator"      # Validates data quality (discrepancy, approvals)
    AGGREGATOR = "aggregator"    # Combines results (quote comparison)


# Mapping of capabilities to roles for automatic workflow composition
CAPABILITY_ROLES: Dict[AgentCapability, AgentRole] = {
    AgentCapability.DOCUMENT_EXTRACTION: AgentRole.SOURCE,
    AgentCapability.DOCUMENT_CLASSIFICATION: AgentRole.SOURCE,
    AgentCapability.OPPORTUNITY_MINING: AgentRole.SOURCE,
    AgentCapability.SUPPLIER_RANKING: AgentRole.PROCESSOR,
    AgentCapability.QUOTE_EVALUATION: AgentRole.PROCESSOR,
    AgentCapability.QUOTE_COMPARISON: AgentRole.AGGREGATOR,
    AgentCapability.EMAIL_DRAFTING: AgentRole.COMMUNICATOR,
    AgentCapability.EMAIL_DISPATCH: AgentRole.COMMUNICATOR,
    AgentCapability.EMAIL_WATCHING: AgentRole.COMMUNICATOR,
    AgentCapability.NEGOTIATION: AgentRole.COMMUNICATOR,
    AgentCapability.SUPPLIER_INTERACTION: AgentRole.COMMUNICATOR,
    AgentCapability.DISCREPANCY_DETECTION: AgentRole.VALIDATOR,
    AgentCapability.APPROVAL_DECISION: AgentRole.VALIDATOR,
    AgentCapability.RAG_QUERY: AgentRole.PROCESSOR,
}
