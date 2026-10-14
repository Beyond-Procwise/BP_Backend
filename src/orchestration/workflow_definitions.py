"""Declarative workflow definitions for the ProcWise procurement system.

Each function returns a :class:`WorkflowGraph` that can be executed by the
:class:`WorkflowEngine`.  Workflows are composed of agent nodes and edges
with explicit data flow mappings.

This replaces the hardcoded ``_execute_*_workflow`` methods in the monolithic
orchestrator with composable, testable, and reusable workflow definitions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from orchestration.workflow_engine import (
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
    WorkflowState,
    NodeStatus,
)


# ---------------------------------------------------------------------------
# Derived-value helpers (mirror the legacy orchestrator so the declarative path
# carries the same context downstream)
# ---------------------------------------------------------------------------
def _derive_product_category(opportunity_payload: Any) -> Optional[str]:
    """Pick the dominant spend category from an opportunity payload.

    Mirrors ``Orchestrator._derive_product_category`` so downstream nodes
    (quote evaluation, email drafting) receive category context on the
    declarative path exactly as they did on the legacy path.
    """
    if not isinstance(opportunity_payload, dict):
        return None

    def _norm(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    for key in ("product_category", "category_id", "primary_category", "spend_category"):
        direct = _norm(opportunity_payload.get(key))
        if direct:
            return direct

    findings = opportunity_payload.get("findings")
    if isinstance(findings, list):
        totals: Dict[str, float] = {}
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            category = _norm(
                finding.get("category_id")
                or finding.get("spend_category")
                or finding.get("category")
                or finding.get("item_category")
            )
            if not category:
                continue
            weight = finding.get("financial_impact_gbp")
            if weight in (None, ""):
                for alt in ("potential_savings", "total_savings",
                            "estimated_savings", "value", "impact"):
                    weight = finding.get(alt)
                    if weight not in (None, ""):
                        break
            try:
                score = float(weight)
            except (TypeError, ValueError):
                score = 0.0
            if score <= 0.0:
                score = 1.0
            totals[category] = totals.get(category, 0.0) + score
        if totals:
            return max(totals.items(), key=lambda item: (item[1], item[0]))[0]

    return None


def _opportunity_post_process(
    result_data: Dict[str, Any], shared_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Derive product_category from the opportunity findings for downstream nodes."""
    payload: Dict[str, Any] = dict(shared_data or {})
    payload.update(result_data or {})
    category = _derive_product_category(payload)
    return {"product_category": category} if category else {}


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


def _requirement_is_complete(state: WorkflowState) -> bool:
    """True once the requirements agent has fully gathered a requirement.

    Gates the hand-off to supplier_ranking: while the agent is still asking
    questions (complete=False) the workflow does not advance to sourcing.
    """
    return bool(
        state.shared_data.get("complete")
        or state.node_results.get("gather_requirement", {}).get("complete")
    )


def _negotiation_fields_present(state: WorkflowState) -> bool:
    required = {"supplier", "current_offer", "target_price", "rfq_id"}
    return required.issubset(state.shared_data.keys())


def _has_drafts(state: WorkflowState) -> bool:
    return bool(state.shared_data.get("drafts"))


def _has_responses(state: WorkflowState) -> bool:
    # `supplier_responses`, not `responses`: that is the key EmailWatcherAgent
    # returns (email_watcher_agent.py:572) and the key
    # NegotiationAgent._extract_batch_inputs reads. This gate asked for
    # `responses`, which nothing produces, so it was permanently False and the
    # negotiate node had never executed in this graph.
    return bool(state.shared_data.get("supplier_responses"))


# ---------------------------------------------------------------------------
# Workflow: Document Extraction
# ---------------------------------------------------------------------------

def _has_extracted_docs(state: WorkflowState) -> bool:
    """Only run discrepancy detection when extraction actually produced documents."""
    payload = state.node_results.get("extract_documents", {}) or {}
    return bool(
        payload.get("extracted_docs")
        or payload.get("processing_issues")
        or payload.get("details")
    )


def build_extraction_workflow() -> WorkflowGraph:
    """Workflow: Extract structured data from procurement documents.

    Graph:
        extract_documents -> detect_discrepancies   (when documents were produced)

    The discrepancy node is now REAL. This docstring has claimed it since the graph was
    written, but no node was ever added: DataExtractionAgent instead hard-instantiated
    DiscrepancyDetectionAgent inside itself (data_extraction_agent.py), bypassing the
    registry, the blackboard and the policy gate. It is an orchestrated node like any other.
    """
    graph = WorkflowGraph(
        name="document_extraction",
        description="Extract and validate structured data from procurement documents",
    )

    graph.add_node(WorkflowNode(
        name="extract_documents",
        agent_type="data_extraction",
        output_to_shared=["details", "summary", "extracted_docs", "processing_issues"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="detect_discrepancies",
        agent_type="discrepancy_detection",
        input_mapping={
            "extract_documents.extracted_docs": "extracted_docs",
            "extract_documents.processing_issues": "processing_issues",
        },
        output_to_shared=["mismatches", "summary"],
        required=False,   # a detection failure must not fail the extraction
    ))

    graph.add_edge(
        "extract_documents", "detect_discrepancies",
        condition=_has_extracted_docs,
        label="documents_extracted",
    )

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
        post_process=_opportunity_post_process,
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
        output_to_shared=["supplier_responses"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="negotiate",
        agent_type="negotiation",
        input_mapping={"watch_responses.supplier_responses": "supplier_responses"},
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


def build_requirements_to_ranking_workflow() -> WorkflowGraph:
    """Workflow: gather a procurement requirement, then drive sourcing.

    Graph::

        gather_requirement --(requirement complete)--> rank_suppliers
                                                            |
                                                  (ranking ready)
                                                            v
                                                       draft_emails

    The requirements agent runs first; only once it reports ``complete`` does
    the workflow hand the requirement off to supplier_ranking (the agent emits
    a ``query`` field for that purpose). RFQ drafting follows when a ranking is
    produced. While the requirement is still being gathered the gate stays shut,
    so an incomplete conversation never triggers sourcing.
    """
    graph = WorkflowGraph(
        name="requirements_to_ranking",
        description="Gather a procurement requirement, then rank suppliers and draft RFQs",
    )

    graph.add_node(WorkflowNode(
        name="gather_requirement",
        agent_type="requirements",
        static_inputs={"created_by": "workflow"},
        output_to_shared=[
            "requirement", "requirement_id", "completeness_score", "complete", "query",
        ],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="rank_suppliers",
        agent_type="supplier_ranking",
        input_mapping={
            "gather_requirement.query": "query",
            "gather_requirement.requirement": "requirement",
        },
        output_to_shared=["ranking", "justification"],
        required=True,
    ))

    graph.add_node(WorkflowNode(
        name="draft_emails",
        agent_type="email_drafting",
        input_mapping={
            "rank_suppliers.ranking": "ranking",
            "gather_requirement.requirement": "requirement",
        },
        output_to_shared=["drafts"],
        required=False,
    ))

    graph.add_edge(
        "gather_requirement", "rank_suppliers",
        condition=_requirement_is_complete,
        label="requirement_complete",
    )
    graph.add_edge(
        "rank_suppliers", "draft_emails",
        condition=_has_ranking_payload,
        label="ranking_ready",
    )

    return graph


# ---------------------------------------------------------------------------
# Workflow: Approvals
# ---------------------------------------------------------------------------

def build_approvals_workflow() -> WorkflowGraph:
    """Workflow: gate a spend amount against the governed approval threshold.

    Graph:
        decide_approval

    A single-node graph, deliberately. `approvals` was absent from the registry
    entirely, so `POST /workflows/approvals` fell through to
    `_execute_generic_workflow` — a routing-rule chain that is not the declarative
    path the rest of the system uses. Registering it here means the approvals
    endpoint runs through the same WorkflowEngine, gets the same audit row in
    proc.bp_agent_actions, and the same governance envelope as every other agent.
    """
    graph = WorkflowGraph(
        name="approvals",
        description="Gate a spend amount against the governed approval threshold",
    )

    graph.add_node(WorkflowNode(
        name="decide_approval",
        agent_type="approvals",
        output_to_shared=["decision", "approved", "threshold", "grounding", "approval_id"],
        required=True,
    ))

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
    "requirements_to_ranking": build_requirements_to_ranking_workflow,
    "approvals": build_approvals_workflow,
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
