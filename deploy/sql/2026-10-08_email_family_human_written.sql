-- Email family: human_written (an email a PERSON typed: the Action Centre reply panel, the report email
-- panel, and the manual passthrough of the drafting agent).
--
-- NOT applied anywhere. bp_testdb first, bp_sqldb only after live verification (ruling 2026-10-08).
-- Insert only; touches no existing row. mode 'shadow': every check is RECORDED, nothing is blocked.
--
-- No model writes these, so there is nothing to repair and nothing to judge. The person is the author, so
-- every figure in their text is CARRIED (accepted, never treated as a fact) and any that does not equal a
-- Postgres value is listed under unverified_figures for the reviewer. What Postgres adds is the other side
-- of the comparison: the supplier's latest offer and currency, and who the contact is. What this family
-- CAN fail is what no author should send: bank details, an admission of liability, a waiver, an award
-- commitment, a leaked internal limit. And the recipient check: an address not on the supplier master.

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT
 'EmailFamily_human_written', 'email_family',
 'Guardrails for an email a person wrote; their figures are reported unverified against the supplier''s own offer.',
 $json${
  "policy_identifier": "email_family_human_written",
  "required_role": "Admin",
  "rules": {
    "family_id": "human_written",
    "mode": "shadow",
    "classifiable": false,
    "length_target": 0,
    "required_facts": [],
    "carried_keys": ["body"],
    "fact_sources": {
      "supplier_current_offer": {"table": "supplier_response", "column": "price", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "number", "label": "Supplier's latest offer"},
      "currency": {"table": "supplier_response", "column": "currency", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "text", "label": "Offer currency"},
      "supplier_contact_name": {"table": "bp_supplier", "column": "contact_name_1", "row_id": "supplier_id",
        "lookup": {"supplier_id": "supplier_id"}, "value_type": "text", "label": "Supplier contact"}
    },
    "context": [],
    "reasoned": {},
    "never_state": {
      "walkaway_price": ["walkaway_price"],
      "market_floor_price": ["market_floor_price"],
      "budget_ceiling": ["budget_ceiling"]
    },
    "required_elements": [],
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
  SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailFamily_human_written');
