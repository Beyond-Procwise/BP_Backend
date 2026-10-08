-- Email family: rfq_batch (a request for quotation sent to each supplier in a ranked list).
--
-- NOT applied anywhere. bp_testdb first, bp_sqldb only after live verification (ruling 2026-10-08).
-- Insert only; touches no existing row. mode 'shadow': every check is RECORDED, nothing is blocked.
--
-- An RFQ states what the BUYER is asking for: a deadline, the items and quantities wanted. Those come from
-- the request and from upstream agents' supplier profiles, not from a Postgres row that could confirm them,
-- so they are carried and every figure that reaches the email is reported under unverified_figures (the
-- spec's "user_asserted"). What Postgres CAN confirm is who the supplier contact is. No offer, price or
-- commitment is ever a fact of an RFQ, so a price in an RFQ body that no payload key carries is a violation.

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT
 'EmailFamily_rfq_batch', 'email_family',
 'Guardrails for the request-for-quotation batch; the buyer''s own figures are reported unverified.',
 $json${
  "policy_identifier": "email_family_rfq_batch",
  "required_role": "Admin",
  "rules": {
    "family_id": "rfq_batch",
    "mode": "shadow",
    "classifiable": false,
    "length_target": 220,
    "required_facts": [],
    "carried_keys": ["deadline", "submission_deadline", "supplier_profile", "line_items", "asks", "scope"],
    "fact_sources": {
      "supplier_contact_name": {"table": "bp_supplier", "column": "contact_name_1", "row_id": "supplier_id",
        "lookup": {"supplier_id": "supplier_id"}, "value_type": "text", "label": "Supplier contact",
        "caller_keys": ["contact_name", "supplier_contact"]}
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
  SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailFamily_rfq_batch');
