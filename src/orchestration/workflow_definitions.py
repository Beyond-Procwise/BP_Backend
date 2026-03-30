"""Declarative workflow definitions for the ProcWise procurement system.

Each function returns a :class:`WorkflowGraph` that can be executed by the
:class:`WorkflowEngine`.  Workflows are composed of agent nodes and edges
with explicit data flow mappings.

This replaces the hardcoded ``_execute_*_workflow`` methods in the monolithic
orchestrator with composable, testable, and reusable workflow definitions.
"""

from __future__ import annotations

from orchestration.workflow_engine import (
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
    WorkflowState,
    NodeStatus,
)


# ---------------------------------------------------------------------------
# Condition helpers for edge traversal
# ---------------------------------------------------------------------------

def _extraction_succeeded(state: WorkflowState) -> bool:
    return state.node_statuses.get("extract_documents") == NodeStatus.COMPLETED


def _has_supplier_candidates(state: WorkflowState) -> bool:
    candidates = (
        state.shared_data.get("supplier_candidates")
        or state.node_results.get("mine_opportunities", {}).get("supplier_candidates")
    )
    return bool(candidates)


def _ranking_succeeded(state: WorkflowState) -> bool:
    return state.node_statuses.get("rank_suppliers") == NodeStatus.COMPLETED


def _has_ranking_payload(state: WorkflowState) -> bool:
    return bool(
        state.shared_data.get("ranking")
        or state.node_results.get("rank_suppliers", {}).get("ranking")
    )


def _negotiation_fields_present(state: WorkflowState) -> bool:
    required = {"supplier", "current_offer", "target_price", "rfq_id"}
    return required.issubset(state.shared_data.keys())


def _has_drafts(state: WorkflowState) -> bool:
    return bool(state.shared_data.get("drafts"))


def _has_responses(state: WorkflowState) -> bool:
    return bool(state.shared_data.get("responses"))


# ---------------------------------------------------------------------------
# Workflow: Document Extraction
# ---------------------------------------------------------------------------

def build_extraction_workflow() -> WorkflowGraph:
    """Workflow: Extract structured data from procurement documents.

    Graph:
        extract_documents -> [discrepancy_detection]
    """
    graph = WorkflowGraph(
        name="document_extraction",
        description="Extract and validate structured data from procurement documents",
    )

    graph.add_node(WorkflowNode(
        name="extract_documents",
        agent_type="data_extraction",
        output_to_shared=["details", "summary"],
        required=True,
    ))

    return graph


# ---------------------------------------------------------------------------
# Workflow: Supplier Ranking
# ---------------------------------------------------------------------------

def build_ranking_workflow() -> WorkflowGraph:
    """Workflow: Rank suppliers based on policies and performance.

    Graph:
        mine_opportunities -> rank_suppliers -> evaluate_quotes -> draft_emails
    """
    graph = WorkflowGraph(
        name="supplier_ranking",
        description="Discover opportunities, rank suppliers, evaluate quotes, and draft RFQ emails",
    )

    graph.add_node(WorkflowNode(
        name="mine_opportunities",
        agent_type="opportunity_miner",
        output_to_shared=["findings", "supplier_candidates", "supplier_directory"],
        required=False,
    ))

    graph.add_node(WorkflowNode(
        name="rank_suppliers",
        agent_type="supplier_ranking",
        input_mapping={
            "mine_opportunities.supplier_candidates": "supplier_candidates",
            "mine_opportunities.supplier_directory": "supplier_directory",
        },
        output_to_shared=["ranking"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="evaluate_quotes",
        agent_type="quote_evaluation",
        input_mapping={
            "rank_suppliers.ranking": "ranking",
        },
        output_to_shared=["quotes", "evaluation"],
        required=False,
    ))

    graph.add_node(WorkflowNode(
        name="draft_emails",
        agent_type="email_drafting",
        input_mapping={
            "rank_suppliers.ranking": "ranking",
            "mine_opportunities.findings": "findings",
            "evaluate_quotes.quotes": "quotes",
        },
        output_to_shared=["drafts"],
        required=False,
    ))

    # Edges with conditions
    graph.add_edge(
        "mine_opportunities", "rank_suppliers",
        condition=_has_supplier_candidates,
        label="suppliers_found",
    )
    graph.add_edge(
        "rank_suppliers", "evaluate_quotes",
        condition=_has_ranking_payload,
        label="ranking_ready",
    )
    graph.add_edge(
        "evaluate_quotes", "draft_emails",
        label="quotes_evaluated",
    )

    return graph


# ---------------------------------------------------------------------------
# Workflow: Quote Evaluation
# ---------------------------------------------------------------------------

def build_quote_workflow() -> WorkflowGraph:
    """Workflow: Evaluate quotes and optionally negotiate.

    Graph:
        evaluate_quotes -> [negotiate]
    """
    graph = WorkflowGraph(
        name="quote_evaluation",
        description="Evaluate supplier quotes and trigger negotiation if needed",
    )

    graph.add_node(WorkflowNode(
        name="evaluate_quotes",
        agent_type="quote_evaluation",
        output_to_shared=["quotes", "evaluation", "supplier", "current_offer", "target_price", "rfq_id"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="negotiate",
        agent_type="negotiation",
        required=False,
    ))

    graph.add_edge(
        "evaluate_quotes", "negotiate",
        condition=_negotiation_fields_present,
        label="negotiation_needed",
    )

    return graph


# ---------------------------------------------------------------------------
# Workflow: Opportunity Mining (Full Pipeline)
# ---------------------------------------------------------------------------

def build_opportunity_workflow() -> WorkflowGraph:
    """Workflow: Full opportunity-to-RFQ pipeline.

    Graph:
        mine_opportunities -> rank_suppliers -> evaluate_quotes -> draft_emails
    """
    graph = WorkflowGraph(
        name="opportunity_mining",
        description="End-to-end: discover opportunities, rank, evaluate, and draft RFQs",
    )

    graph.add_node(WorkflowNode(
        name="mine_opportunities",
        agent_type="opportunity_miner",
        output_to_shared=["findings", "supplier_candidates", "supplier_directory"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="rank_suppliers",
        agent_type="supplier_ranking",
        input_mapping={
            "mine_opportunities.supplier_candidates": "supplier_candidates",
            "mine_opportunities.supplier_directory": "supplier_directory",
        },
        output_to_shared=["ranking"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="evaluate_quotes",
        agent_type="quote_evaluation",
        input_mapping={
            "rank_suppliers.ranking": "ranking",
        },
        output_to_shared=["quotes", "evaluation"],
        required=False,
    ))

    graph.add_node(WorkflowNode(
        name="draft_emails",
        agent_type="email_drafting",
        input_mapping={
            "rank_suppliers.ranking": "ranking",
            "mine_opportunities.findings": "findings",
            "evaluate_quotes.quotes": "quotes",
        },
        output_to_shared=["drafts"],
        required=False,
    ))

    graph.add_edge(
        "mine_opportunities", "rank_suppliers",
        condition=_has_supplier_candidates,
        label="suppliers_found",
    )
    graph.add_edge(
        "rank_suppliers", "evaluate_quotes",
        condition=_has_ranking_payload,
        label="ranking_ready",
    )
    graph.add_edge(
        "evaluate_quotes", "draft_emails",
        label="quotes_evaluated",
    )

    return graph


# ---------------------------------------------------------------------------
# Workflow: Supplier Interaction (Multi-round Negotiation)
# ---------------------------------------------------------------------------

def build_supplier_interaction_workflow() -> WorkflowGraph:
    """Workflow: Draft -> Dispatch -> Watch -> Negotiate cycle.

    Graph:
        draft_emails -> dispatch_emails -> watch_responses -> negotiate -> compare_quotes
    """
    graph = WorkflowGraph(
        name="supplier_interaction",
        description="Multi-round supplier communication: draft, send, watch, negotiate, compare",
    )

    graph.add_node(WorkflowNode(
        name="draft_emails",
        agent_type="email_drafting",
        output_to_shared=["drafts"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="dispatch_emails",
        agent_type="email_dispatch",
        input_mapping={"draft_emails.drafts": "drafts"},
        output_to_shared=["dispatch_results"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="watch_responses",
        agent_type="email_watcher",
        output_to_shared=["responses"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="negotiate",
        agent_type="negotiation",
        input_mapping={"watch_responses.responses": "responses"},
        output_to_shared=["negotiation_result"],
        required=False,
    ))

    graph.add_node(WorkflowNode(
        name="compare_quotes",
        agent_type="quote_comparison",
        output_to_shared=["comparison", "recommendation"],
        required=False,
    ))

    graph.add_edge("draft_emails", "dispatch_emails", condition=_has_drafts)
    graph.add_edge("dispatch_emails", "watch_responses")
    graph.add_edge("watch_responses", "negotiate", condition=_has_responses)
    graph.add_edge("negotiate", "compare_quotes")

    return graph


# ---------------------------------------------------------------------------
# Workflow Registry
# ---------------------------------------------------------------------------

WORKFLOW_REGISTRY = {
    "document_extraction": build_extraction_workflow,
    "supplier_ranking": build_ranking_workflow,
    "quote_evaluation": build_quote_workflow,
    "opportunity_mining": build_opportunity_workflow,
    "supplier_interaction": build_supplier_interaction_workflow,
}


def get_workflow(name: str) -> WorkflowGraph:
    """Retrieve a workflow graph by name."""
    builder = WORKFLOW_REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown workflow '{name}'. Available: {sorted(WORKFLOW_REGISTRY.keys())}"
        )
    return builder()


def list_workflows() -> list[str]:
    """Return all available workflow names."""
    return sorted(WORKFLOW_REGISTRY.keys())
