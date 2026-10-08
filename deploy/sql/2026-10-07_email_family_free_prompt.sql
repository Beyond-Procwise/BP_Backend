-- Email family: free_prompt (an email composed from a person's own words).
--
-- Applied to bp_testdb only. Insert only; touches no existing row. mode 'shadow'.
--
-- There is no fact table to read here: the figures in the draft come from what the
-- person typed. Per the spec those are not facts. They are accepted as carried and
-- every one that reaches the email is listed under unverified_figures for review.

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT
 'EmailFamily_free_prompt', 'email_family',
 'Guardrails for an email composed from free text; figures from the prompt are reported unverified.',
 $json${
  "policy_identifier": "email_family_free_prompt",
  "required_role": "Admin",
  "rules": {
    "family_id": "free_prompt",
    "mode": "shadow",
    "length_target": 300,
    "required_facts": [],
    "carried_keys": ["prompt", "counter_price", "counter_proposals", "deadline", "asks", "line_items"],
    "fact_sources": {
      "supplier_contact_name": {"table": "bp_supplier", "column": "contact_name_1", "row_id": "supplier_id",
        "lookup": {"supplier_id": "supplier_id"}, "value_type": "text",
        "caller_keys": ["contact_name", "supplier_contact"]}
    },
    "context": ["email_thread_summary"],
    "reasoned": {},
    "never_state": {
      "walkaway_price": ["walkaway_price"],
      "market_floor_price": ["market_floor_price"]
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
  SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailFamily_free_prompt');
