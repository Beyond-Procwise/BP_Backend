"""Agent factory for creating and managing agent instances.

The factory pattern decouples agent creation from the orchestrator, making
agents independently testable and reusable across workflows.  Agents are
registered by their canonical type and can be instantiated with different
configurations.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from agents.agent_interface import AgentCapability, AgentContract, AgentInterface
from agents.base_agent import BaseAgent
from agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

# Canonical mapping from agent type slugs to module paths
_AGENT_MODULE_MAP: Dict[str, str] = {
    "data_extraction": "agents.data_extraction_agent",
    "supplier_ranking": "agents.supplier_ranking_agent",
    "quote_comparison": "agents.quote_comparison_agent",
    "quote_evaluation": "agents.quote_evaluation_agent",
    "opportunity_miner": "agents.opportunity_miner_agent",
    "email_drafting": "agents.email_drafting_agent",
    "email_dispatch": "agents.email_dispatch_agent",
    "email_watcher": "agents.email_watcher_agent",
    "negotiation": "agents.negotiation_agent",
    "supplier_interaction": "agents.supplier_interaction_agent",
    "approvals": "agents.approvals_agent",
    "discrepancy_detection": "agents.discrepancy_detection_agent",
    "rag": "agents.rag_agent",
}

# Canonical mapping from agent type slugs to class names
_AGENT_CLASS_MAP: Dict[str, str] = {
    "data_extraction": "DataExtractionAgent",
    "supplier_ranking": "SupplierRankingAgent",
    "quote_comparison": "QuoteComparisonAgent",
    "quote_evaluation": "QuoteEvaluationAgent",
    "opportunity_miner": "OpportunityMinerAgent",
    "email_drafting": "EmailDraftingAgent",
    "email_dispatch": "EmailDispatchAgent",
    "email_watcher": "EmailWatcherAgent",
    "negotiation": "NegotiationAgent",
    "supplier_interaction": "SupplierInteractionAgent",
    "approvals": "ApprovalsAgent",
    "discrepancy_detection": "DiscrepancyDetectionAgent",
    "rag": "RAGAgent",
}

# Agent contracts define each agent's I/O specification
AGENT_CONTRACTS: Dict[str, AgentContract] = {
    "data_extraction": AgentContract(
        agent_type="data_extraction",
        capabilities=frozenset({
            AgentCapability.DOCUMENT_EXTRACTION,
            AgentCapability.DOCUMENT_CLASSIFICATION,
            AgentCapability.HEADER_EXTRACTION,
            AgentCapability.LINE_ITEM_EXTRACTION,
        }),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"s3_prefix", "s3_object_key"}),
        output_fields=frozenset({"details", "summary", "mismatches"}),
        description="Extracts structured data from procurement documents (PDF, DOCX, images)",
    ),
    "supplier_ranking": AgentContract(
        agent_type="supplier_ranking",
        capabilities=frozenset({AgentCapability.SUPPLIER_RANKING}),
        required_inputs=frozenset({"query"}),
        optional_inputs=frozenset({"criteria", "supplier_data", "supplier_candidates"}),
        output_fields=frozenset({"ranking", "justification"}),
        description="Ranks suppliers based on policies and performance data",
    ),
    "quote_comparison": AgentContract(
        agent_type="quote_comparison",
        capabilities=frozenset({AgentCapability.QUOTE_COMPARISON}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"quotes", "ranking"}),
        output_fields=frozenset({"comparison", "recommendation"}),
        description="Aggregates and compares supplier quotes",
    ),
    "quote_evaluation": AgentContract(
        agent_type="quote_evaluation",
        capabilities=frozenset({AgentCapability.QUOTE_EVALUATION}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"ranking", "supplier_candidates"}),
        output_fields=frozenset({"quotes", "evaluation"}),
        description="Evaluates supplier quotes against policies and historical data",
    ),
    "opportunity_miner": AgentContract(
        agent_type="opportunity_miner",
        capabilities=frozenset({AgentCapability.OPPORTUNITY_MINING}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"query", "product_category"}),
        output_fields=frozenset({"findings", "supplier_candidates", "supplier_directory"}),
        description="Identifies procurement anomalies and savings opportunities",
    ),
    "email_drafting": AgentContract(
        agent_type="email_drafting",
        capabilities=frozenset({AgentCapability.EMAIL_DRAFTING}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"ranking", "findings", "negotiation_context"}),
        output_fields=frozenset({"drafts"}),
        description="Drafts communication emails for suppliers",
    ),
    "email_dispatch": AgentContract(
        agent_type="email_dispatch",
        capabilities=frozenset({AgentCapability.EMAIL_DISPATCH}),
        required_inputs=frozenset({"drafts"}),
        optional_inputs=frozenset(),
        output_fields=frozenset({"dispatch_results"}),
        description="Sends approved email drafts to suppliers",
    ),
    "email_watcher": AgentContract(
        agent_type="email_watcher",
        capabilities=frozenset({AgentCapability.EMAIL_WATCHING}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"workflow_id"}),
        output_fields=frozenset({"responses"}),
        description="Monitors inbound supplier email responses",
    ),
    "negotiation": AgentContract(
        agent_type="negotiation",
        capabilities=frozenset({AgentCapability.NEGOTIATION}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({
            "supplier", "current_offer", "target_price", "rfq_id",
            "negotiation_context",
        }),
        output_fields=frozenset({"negotiation_result", "drafts"}),
        description="Generates multi-round supplier negotiation messages",
    ),
    "supplier_interaction": AgentContract(
        agent_type="supplier_interaction",
        capabilities=frozenset({AgentCapability.SUPPLIER_INTERACTION}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"draft_payload", "supplier_input"}),
        output_fields=frozenset({"interaction_result"}),
        description="Normalizes and routes supplier communications",
    ),
    "approvals": AgentContract(
        agent_type="approvals",
        capabilities=frozenset({AgentCapability.APPROVAL_DECISION}),
        required_inputs=frozenset(),
        optional_inputs=frozenset({"policy_context"}),
        output_fields=frozenset({"decision", "justification"}),
        description="Determines approval decisions based on policy thresholds",
    ),
    "discrepancy_detection": AgentContract(
        agent_type="discrepancy_detection",
        capabilities=frozenset({AgentCapability.DISCREPANCY_DETECTION}),
        required_inputs=frozenset({"extracted_docs"}),
        optional_inputs=frozenset({"processing_issues"}),
        output_fields=frozenset({"mismatches"}),
        description="Detects data quality issues and mismatches",
    ),
    "rag": AgentContract(
        agent_type="rag",
        capabilities=frozenset({AgentCapability.RAG_QUERY}),
        required_inputs=frozenset({"query"}),
        optional_inputs=frozenset(),
        output_fields=frozenset({"answer", "sources"}),
        description="Retrieval-augmented generation for procurement Q&A",
    ),
}


def _slugify(name: str) -> str:
    """Convert CamelCase or mixed identifiers to snake_case slugs."""
    text = re.sub(r"(?<!^)(?=[A-Z][a-z0-9])", "_", str(name).strip())
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).lower().strip("_")
    # Remove trailing '_agent' for canonical slugs
    if slug.endswith("_agent"):
        slug = slug[: -len("_agent")]
    return slug


class AgentFactory:
    """Creates and manages agent instances with proper dependency injection.

    The factory loads agent definitions from ``agent_definitions.json``,
    resolves the Python module and class for each agent type, and provides
    methods to create instances with the required dependencies.

    Usage::

        factory = AgentFactory(agent_nick)
        agent = factory.create("data_extraction")
        result = agent.execute(context)

        # Or build all agents at once
        registry = factory.build_registry()
    """

    def __init__(self, agent_nick: Any) -> None:
        self._agent_nick = agent_nick
        self._definitions = self._load_definitions()
        self._instances: Dict[str, BaseAgent] = {}

    def _load_definitions(self) -> List[Dict[str, Any]]:
        """Load agent definitions from the JSON configuration."""
        paths = [
            Path(__file__).resolve().parent.parent / "agent_definitions.json",
            Path(__file__).resolve().parent.parent.parent / "agent_definitions.json",
        ]
        for path in paths:
            if path.exists():
                try:
                    with path.open() as f:
                        return json.load(f)
                except Exception:
                    logger.warning("Failed to load agent definitions from %s", path)
        return []

    def create(self, agent_type: str) -> BaseAgent:
        """Create or retrieve a cached agent instance by type slug."""
        slug = _slugify(agent_type)
        if slug in self._instances:
            return self._instances[slug]

        module_path = _AGENT_MODULE_MAP.get(slug)
        class_name = _AGENT_CLASS_MAP.get(slug)
        if not module_path or not class_name:
            raise ValueError(
                f"Unknown agent type '{agent_type}' (slug: '{slug}'). "
                f"Available: {sorted(_AGENT_MODULE_MAP.keys())}"
            )

        try:
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Failed to import {class_name} from {module_path}: {exc}"
            ) from exc

        instance = agent_class(self._agent_nick)
        self._instances[slug] = instance
        return instance

    def get_contract(self, agent_type: str) -> Optional[AgentContract]:
        """Return the contract for a given agent type."""
        slug = _slugify(agent_type)
        return AGENT_CONTRACTS.get(slug)

    def build_registry(self) -> AgentRegistry:
        """Build a complete agent registry with all defined agents.

        Returns an ``AgentRegistry`` populated with agent instances keyed
        by their canonical slug, with CamelCase aliases registered for
        backward compatibility.
        """
        registry = AgentRegistry()
        aliases: Dict[str, str] = {}

        for slug in _AGENT_MODULE_MAP:
            try:
                agent = self.create(slug)
                registry[slug] = agent
                # Register CamelCase alias
                class_name = _AGENT_CLASS_MAP.get(slug, "")
                if class_name:
                    aliases[class_name] = slug
            except Exception as exc:
                logger.warning("Failed to create agent '%s': %s", slug, exc)

        registry.add_aliases(aliases)
        return registry

    def available_agents(self) -> List[str]:
        """Return the list of available agent type slugs."""
        return sorted(_AGENT_MODULE_MAP.keys())

    def get_capabilities(self, agent_type: str) -> Optional[frozenset]:
        """Return the capabilities for a given agent type."""
        contract = self.get_contract(agent_type)
        return contract.capabilities if contract else None
