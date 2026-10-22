"""Which proc.* tables belong to a customer (tenant), which are the platform's, and which need a decision.

This is the single list the tenant-isolation migrations are generated from (backlog 1.19, D-37).
Every table must be in exactly ONE group, and a table that is in none makes the generator and
tests/tenancy refuse to run -- so a new table cannot quietly ship without an isolation decision.

  OWNED   customer data. Gets a ``tenant_id`` label, a forced row-level-security policy.
  SHARED  platform definitions every customer reads the same way (prompts, models, FX rates,
          translations, rule/policy definitions). No label. A customer-specific override would
          need its own label later; that is a separate decision.
  REVIEW  could be either (learned extraction patterns, taxonomies, admin config, tables that
          already carry a ``tenant_id`` meaning "NULL = global"). NOT touched by any generated
          migration until a person moves it to OWNED or SHARED. Empty since D-40 (2026-10-21,
          user: recommended): 5 moved to SHARED, 8 to OWNED. A table added later starts here.
"""
from __future__ import annotations

OWNED = frozenset({
    # messaging module (exists on bp_testdb only so far; D-56)
    "bp_conversation", "bp_conversation_participant", "bp_domain_event", "bp_message",
    "bp_notification", "bp_person", "bp_routing_rule",
    "agent_plan", "bp_account", "bp_account_contact",
    "bp_account_history_scope", "bp_account_price", "bp_action",
    "bp_admin_config", "bp_agent_actions", "bp_agent_definition",
    "bp_agent_definition_version", "bp_agent_group", "bp_agent_policy",
    "bp_agent_policy_conflict", "bp_agent_policy_conflict_rule", "bp_agent_policy_person",
    "bp_agent_policy_second_review", "bp_agent_policy_suggestion", "bp_agent_policy_version",
    "bp_agent_run", "bp_agent_workflow", "bp_analysis",
    "bp_analysis_deal", "bp_analysis_document", "bp_analysis_summary",
    "bp_approval", "bp_artifact_derivation", "bp_brand_logo",
    "bp_business_area", "bp_catalog_cost_tier", "bp_catalog_item",
    "bp_catalog_item_match", "bp_catalog_item_relation", "bp_catalog_mapping",
    "bp_catalog_source", "bp_catalog_special_bid", "bp_category",
    "bp_commercial_fact", "bp_concept", "bp_constraint",
    "bp_contract_expiry_alert", "bp_contract_line_items", "bp_contract_line_items_raw",
    "bp_contract_obligation", "bp_contract_obligation_party", "bp_contract_obligation_run",
    "bp_contract_raw", "bp_contracts", "bp_deal",
    "bp_deal_document_map", "bp_deal_proposal", "bp_deal_proposal_member",
    "bp_decision", "bp_demand", "bp_detection_finding",
    "bp_doc_number_counter", "bp_egress_event", "bp_embedding_job",
    "bp_extraction_discrepancy", "bp_extraction_discrepancy_bkp", "bp_extraction_hallucination_audit",
    "bp_extraction_health_metrics", "bp_extraction_hint_proposal", "bp_extraction_observation",
    "bp_extraction_provenance", "bp_extraction_provenance_v3", "bp_extraction_retry_log",
    "bp_extraction_telemetry", "bp_extraction_template", "bp_extraction_verdict",
    "bp_fact_provenance", "bp_finding_fact", "bp_gateway_call",
    "bp_gateway_write", "bp_goods_receipt_line_items_raw", "bp_goods_receipt_line_items_stg",
    "bp_goods_receipt_line_items_trgt", "bp_goods_receipt_raw", "bp_goods_receipt_stg",
    "bp_goods_receipt_trgt", "bp_ingest_arrival", "bp_ingest_artifact",
    "bp_ingest_bundle", "bp_ingest_sender_auth", "bp_internal_email",
    "bp_invoice_line_items_raw", "bp_invoice_line_items_stg", "bp_invoice_line_items_trgt",
    "bp_invoice_line_items_trgt_bkp", "bp_invoice_raw", "bp_invoice_raw_bkp",
    "bp_invoice_raw_bkp_june12", "bp_invoice_stg", "bp_invoice_stg_bkp_june12",
    "bp_invoice_stg_trgt_june12", "bp_invoice_trgt", "bp_invoice_trgt_bkp",
    "bp_link_rejection", "bp_mailbox_binding", "bp_negotiation_advice",
    "bp_negotiation_advice_fact", "bp_opportunity", "bp_opportunity_critique",
    "bp_opportunity_gap", "bp_page_layout", "bp_playbook",
    "bp_playbook_proposal", "bp_po_line_items_raw", "bp_po_line_items_stg",
    "bp_po_line_items_trgt", "bp_po_line_items_trgt_bkp", "bp_po_raw_june12",
    "bp_po_stg_june12", "bp_po_trgt_june12", "bp_policy_document",
    "bp_policy_document_version", "bp_policy_extraction_item", "bp_policy_extraction_run",
    "bp_policy_firing", "bp_policy_firing_reversal", "bp_policy_notification",
    "bp_policy_observation", "bp_presentation_log", "bp_prompt_render",
    "bp_prompt_render_use", "bp_purchase_order_raw", "bp_purchase_order_stg",
    "bp_purchase_order_trgt", "bp_purchase_order_trgt_bkp", "bp_quote_evaluation",
    "bp_quote_line_items_raw", "bp_quote_line_items_stg", "bp_quote_line_items_trgt",
    "bp_quote_line_items_trgt_bkp", "bp_quote_raw", "bp_quote_raw_june12",
    "bp_quote_stg", "bp_quote_stg_june12", "bp_quote_trgt",
    "bp_quote_trgt_bkp", "bp_quote_trgt_june12", "bp_report_job",
    "bp_report_version", "bp_reports", "bp_requirement",
    "bp_role_assignment", "bp_sales_justification", "bp_sales_opportunity",
    "bp_sales_quote", "bp_sales_quote_internal", "bp_sales_quote_line",
    "bp_sales_quote_line_internal", "bp_sales_quote_outcome", "bp_sales_quote_template",
    "bp_standin_actor_backfill", "bp_style_divergence", "bp_style_exemplar",
    "bp_style_ingest_staging", "bp_style_intent", "bp_style_pack",
    "bp_style_profile", "bp_style_recompile_suggestion", "bp_summary",
    "bp_supplier", "bp_supplier_alias", "bp_supplier_enrichment",
    "bp_supplier_id_crosswalk", "bp_supplier_info_request", "bp_supplier_name_reject",
    "bp_supplier_ranking", "bp_supplier_review", "bp_support_ticket",
    "bp_tprm_supplier", "bp_triage_deal_state", "bp_triage_finding",
    "bp_triage_result", "bp_triage_run", "bp_user_buyer_scope",
    "bp_user_preference", "bp_user_reseller_link", "bp_value_outcome",
    "bp_vector_chunk_bge_large_v1", "bp_vector_source", "bp_vendor_extraction_profiles",
    "bp_workflow_input_request", "bp_workflow_run", "deal_document_id_map",
    "draft_rfq_emails", "email_dispatch", "email_dispatch_chains",
    "email_thread_map", "extraction_review_queue", "negotiation_session_state",
    "negotiation_sessions", "node_execution", "opportunity_feedback",
    "process_monitor", "process_monitor_bkp", "procurement_flow",
    "rag_feedback", "raw_contracts", "raw_invoice",
    "raw_purchase_order", "raw_quotes", "routing",
    "session_document_outcome", "supplier_response", "supplier_risk_scores",
    "supplier_risk_signals", "workflow_email_tracking", "workflow_events",
    "workflow_execution", "workflow_lifecycle", "workflow_round_responses",
})

SHARED = frozenset({
    "bp_complaince_metric_prty_lkup", "bp_document_type", "bp_extraction_patterns",
    "bp_extraction_type_priors", "bp_fx_rates", "bp_i18n_public_key",
    "bp_lifecycle_transition", "bp_model", "bp_orchestrator_registry",
    "bp_policy", "bp_policy_decider_map", "bp_prompt",
    "bp_prompt_pointer_move", "bp_prompt_registry", "bp_prompt_version",
    "bp_rule", "bp_translation", "bp_translation_language_status",
    "bp_uom_canonical", "procurement_patterns",
})

REVIEW = frozenset({
})
