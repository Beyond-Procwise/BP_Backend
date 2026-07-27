"""Which tables the test dataset must fill, and why.

Every table in both live databases lands in exactly one bucket. The rules are
explicit and testable so that "we cover everything except X" is a claim someone
can check, rather than an assertion.

Buckets:

    BACKUP    Historical or dated copies (_bkp, _june12, invoice1). Filling them
              would misrepresent them as current data.
    OUTPUT    Conclusions the product derives -- rankings, decisions, summaries,
              findings, run logs. The seeder must not write these: anything it
              writes here, no test can afterwards prove.
    REFERENCE Configuration copied verbatim from live, so behaviour matches
              production.
    SEED      Business data the seeder synthesises.
    UNCLEAR   No confident classification. Listed explicitly rather than
              silently skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BACKUP = "BACKUP"
OUTPUT = "OUTPUT"
REFERENCE = "REFERENCE"
SEED = "SEED"
UNCLEAR = "UNCLEAR"

# Dated or duplicated copies. Ordered most-specific first.
_BACKUP_PATTERNS = (
    r"_bkp(_|$)", r"_bkup(_|$)", r"_backup(_|$)", r"_bkp$", r"_old(_|$)", r"_old$",
    r"_june12$", r"_may_4th$", r"_27_04$", r"_180925$", r"_260925$",
    r"_stage$", r"_test$", r"_excel$", r"_new1$", r"_new$", r"_denorm$",
    r"^invoice1$", r"^po1$", r"^agents$", r"^contracts_", r"^contract_agent_",
    r"_bkup$", r"_bkp_", r"^supplier_bkp$", r"^invoice_old$", r"^invoice_new_bkup$",
)

# Things the product concludes. Never seeded.
_OUTPUT_EXACT = frozenset({
    # ranking, evaluation, decision, action, summary
    "bp_supplier_ranking", "bp_quote_evaluation", "bp_decision", "bp_action",
    "bp_summary", "bp_analysis_summary", "bp_reports", "bp_opportunity",
    "bp_detection_finding", "bp_supplier_review", "bp_supplier_enrichment",
    "bp_supplier_name_reject", "bp_supplier_alias",
    "opportunity_feedback", "opportunity_details", "opportunity_uc",
    "marketing_comparison_ui", "office_furniture_comparison_ui",
    # deal formation
    "bp_deal", "bp_deal_document_map", "bp_deal_proposal",
    "bp_deal_proposal_member", "deal_document_id_map",
    # extraction runtime and telemetry
    "bp_extraction_discrepancy", "bp_extraction_hallucination_audit",
    "bp_extraction_health_metrics", "bp_extraction_hint_proposal",
    "bp_extraction_observation", "bp_extraction_provenance",
    "bp_extraction_provenance_v3", "bp_extraction_retry_log",
    "bp_extraction_telemetry", "bp_extraction_template", "bp_extraction_patterns",
    "bp_extraction_type_priors", "extraction_review_queue",
    "bp_data_discrepancies", "bp_discrepancy_data", "data_discrepancy",
    # workflow and agent runtime
    "agent_plan", "bp_agent_actions", "bp_agent_workflow", "bp_workflow_run",
    "bp_workflow_input_request", "node_execution", "workflow_events",
    "workflow_execution", "workflow_lifecycle", "workflow_email_tracking",
    "workflow_round_responses", "workflow_round_watcher", "procurement_flow",
    "process_monitor", "process_run_audit", "session_document_outcome",
    "routing", "routing_run_details", "agent", "agent_training_queue",
    "agent_id_trigger", "category_agent", "contract_agent", "email_agent_uc",
    "invoice_agent", "purchase_order_agent", "quote_agent",
    "invoice_line_items_agent", "po_line_items_agent", "quote_line_items_agent",
    "supplier_interaction_agent", "invoice_agent_stage",
    # negotiation and correspondence
    "negotiation_session_state", "negotiation_sessions", "draft_rfq_emails",
    "email_dispatch", "email_dispatch_chains", "email_thread_map",
    "email_watcher_watermarks", "processed_emails", "rfq_targets",
    "supplier_response", "supplier_interaction", "supplier_relationship_refresh",
    "connections", "bp_support_ticket", "rag_feedback", "rag_training_examples",
    "bp_approval", "bp_approvals", "approvals_uc", "bp_demand",
    "bp_contract_obligation", "bp_contract_obligation_party",
    "bp_contract_obligation_run", "supplier_risk_signals",
    # uicanvas.action is the agent action log, not a to-do list to seed
    "action",
})

_OUTPUT_PATTERNS = (r"_agent$", r"^bp_agent", r"^workflow_", r"^bp_workflow")

# Configuration copied verbatim so the test databases behave like production.
_REFERENCE_EXACT = frozenset({
    "bp_fx_rates", "bp_policy", "bp_prompt", "bp_admin_config",
    "bp_vendor_extraction_profiles", "bp_complaince_metric_prty_lkup",
    "procurement_patterns", "bp_category", "category", "category_mapping",
    "policy", "static_policy", "prompt", "pricing", "pricing_matrix_ranking_policy",
    "quote_weighting", "quote_values", "vendor_profile", "bp_products",
    "bp_category_product_mapping", "cat_product_mapping",
    # Response-style governance and mailbox configuration: these steer how the
    # product behaves, so the test databases must carry production's values.
    "bp_style_profile", "bp_style_intent", "bp_style_exemplar",
    "bp_style_ingest_staging", "bp_mailbox_binding",
})

# Business data the seeder synthesises.
_SEED_EXACT = frozenset({
    # organisation
    "business_unit", "cost_centre",
    # supplier master and the reference data hanging off it
    "bp_supplier", "supplier", "bp_supplier_id_crosswalk", "bp_tprm_supplier",
    "supplier_risk_scores", "esg_data", "contact", "bp_contact",
    # catalogue
    "item", "bp_requirement",
    # documents: bp_sqldb three-tier
    "bp_quote_raw", "bp_quote_stg", "bp_quote_trgt",
    "bp_purchase_order_raw", "bp_purchase_order_stg", "bp_purchase_order_trgt",
    "bp_invoice_raw", "bp_invoice_stg", "bp_invoice_trgt",
    "bp_quote_line_items_raw", "bp_quote_line_items_stg", "bp_quote_line_items_trgt",
    "bp_po_line_items_raw", "bp_po_line_items_stg", "bp_po_line_items_trgt",
    "bp_invoice_line_items_raw", "bp_invoice_line_items_stg",
    "bp_invoice_line_items_trgt",
    "bp_contract_raw", "bp_contracts", "raw_contracts", "raw_invoice",
    "raw_purchase_order", "raw_quotes", "contract", "contracts", "contract_id",
    # documents: uicanvas
    "bp_invoice", "bp_invoice_trgt", "bp_invoice_line_items",
    "bp_invoice_line_items_trgt", "bp_purchase_order", "bp_purchase_order_trgt",
    "bp_po_line_items", "bp_po_line_items_trgt", "bp_quote", "bp_quote_trgt",
    "bp_quote_line_items", "bp_quote_line_items_trgt", "bp_quote_bp",
    "invoice", "invoice_line_items", "purchase_order", "po_line_items",
    "po_items", "quote_uc", "quote_supplier",
    # cross-reference maps between the two conventions
    "sup_mapping", "sup_mapping_inv_new", "po_mapping", "po_mapping_inv_new",
    "po_supplier_mapping", "old_new_supplier_mapping", "inv_mapping_inv_itms_new",
    "item_mapping_inv_itms_new",
})


@dataclass(frozen=True)
class Classification:
    bucket: str
    reason: str


def classify(table: str) -> Classification:
    """Bucket a table by name. Backup wins over everything: a dated copy of a
    seeded table is still a dated copy."""
    name = table.lower()

    for pattern in _BACKUP_PATTERNS:
        if re.search(pattern, name):
            return Classification(BACKUP, "historical or dated copy")

    if name in _OUTPUT_EXACT:
        return Classification(OUTPUT, "the product derives this")
    for pattern in _OUTPUT_PATTERNS:
        if re.search(pattern, name):
            return Classification(OUTPUT, "the product derives this")

    if name in _REFERENCE_EXACT:
        return Classification(REFERENCE, "configuration copied verbatim from live")

    if name in _SEED_EXACT:
        return Classification(SEED, "business data the seeder synthesises")

    return Classification(UNCLEAR, "no confident classification")


def buckets(tables: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        BACKUP: [], OUTPUT: [], REFERENCE: [], SEED: [], UNCLEAR: []
    }
    for table in tables:
        grouped[classify(table).bucket].append(table)
    return grouped
