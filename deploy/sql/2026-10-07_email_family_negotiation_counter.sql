-- Email family: negotiation_counter. Read by src/services/draft_assurance.
--
-- NOT YET APPLIED to either database. Insert only; touches no existing row.
-- mode is 'shadow': the assurance record is stored and shown, nothing is refused.
-- Moving to 'enforce' is a row edit, deliberately a separate decision.

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT
 'EmailFamily_negotiation_counter', 'email_family',
 'Facts, reasoned fields, guardrails and required elements for a negotiation counter email.',
 $json${
  "policy_identifier": "email_family_negotiation_counter",
  "required_role": "Admin",
  "rules": {
    "family_id": "negotiation_counter",
    "request_description": "A request to counter, negotiate or push back on a supplier's price, rate or terms, including asking for a discount, a lower quote or better payment terms.",
    "request_label": "a counter-offer",
    "mode": "shadow",
    "length_target": 250,
    "required_facts": ["supplier_current_offer", "currency"],
    "fact_sources": {
      "supplier_current_offer": {"table": "supplier_response", "column": "price", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "number",
        "caller_keys": ["current_offer_numeric", "current_offer"]},
      "currency": {"table": "supplier_response", "column": "currency", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "text",
        "caller_keys": ["currency", "currency_code"]},
      "supplier_lead_time": {"table": "supplier_response", "column": "lead_time", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "text"},
      "rfq_id": {"table": "supplier_response", "column": "rfq_id", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "text",
        "caller_keys": ["rfq_id", "rfq"]},
      "supplier_contact_name": {"table": "bp_supplier", "column": "contact_name_1", "row_id": "supplier_id",
        "lookup": {"supplier_id": "supplier_id"}, "value_type": "text",
        "caller_keys": ["contact_name", "supplier_contact"]}
    },
    "context": ["supplier_total_spend", "quality_score", "email_thread_summary", "play_recommendations"],
    "reasoned": {
      "counter_price": {"caller_keys": ["counter_price"], "basis_from": ["supplier_current_offer"]},
      "target_price": {"caller_keys": ["target_price"], "basis_from": ["supplier_current_offer"]},
      "response_deadline": {"caller_keys": ["response_deadline", "deadline"], "basis_from": []},
      "lead_time_request": {"caller_keys": ["lead_time_request"], "basis_from": ["supplier_lead_time"]}
    },
    "never_state": {
      "walkaway_price": ["walkaway_price"],
      "market_floor_price": ["market_floor_price"]
    },
    "required_elements": ["explicit_ask", "deadline"],
    "forbidden_patterns": {
      "bank_details": "\\b[A-Z]{2}\\d{2}[A-Z0-9]{11,30}\\b|sort\\s*code|account\\s*number|\\biban\\b|\\bswift\\b|bank\\s+details",
      "liability_admission": "we\\s+(accept|admit|acknowledge)\\s+(full\\s+)?(liability|responsibility|fault)|our\\s+fault|we\\s+are\\s+(liable|at\\s+fault)",
      "rights_waiver": "we\\s+(hereby\\s+)?waive|waive\\s+(any|our|all)\\s+(right|claim)",
      "award_commitment": "we\\s+(will|shall|are\\s+going\\s+to)\\s+(award|place\\s+the\\s+order|issue\\s+a\\s+(po|purchase\\s+order))|guarantee[d]?\\s+(the\\s+)?(volume|order|business)"
    }
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (
  SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailFamily_negotiation_counter');
