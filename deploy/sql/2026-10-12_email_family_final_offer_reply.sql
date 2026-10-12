-- Email family: final_offer_reply (the reply to a supplier who has called their offer final).
--
-- bp_testdb first; bp_sqldb only after live verification (ruling 2026-10-08).
-- Insert only; touches no existing row. mode 'shadow': every check is RECORDED, nothing is blocked.
--
-- The reply is fixed wording written by code (src/services/negotiation/final_offer.py): an acceptance
-- when the offer is within the limit on record, otherwise a holding reply that commits to nothing.
-- The only figure it states is the supplier's own offer, read from the reply row -- a claim until a
-- person confirms it, exactly as for a counter. Our target, walk-away and market floor must never
-- appear in it. award_commitment is the widened pattern of 2026-10-09_email_award_commitment.sql, which
-- updates only the four families that existed then.

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT
 'EmailFamily_final_offer_reply', 'email_family',
 'Guardrails for the fixed-wording reply to a supplier''s final offer; states only the supplier''s own offer.',
 $json${
  "policy_identifier": "email_family_final_offer_reply",
  "required_role": "Admin",
  "rules": {
    "family_id": "final_offer_reply",
    "mode": "shadow",
    "classifiable": false,
    "length_target": 80,
    "required_facts": [],
    "fact_sources": {
      "supplier_current_offer": {"table": "supplier_response", "column": "price", "claim_column": "extraction_status", "claim_unless": ["confirmed"], "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "number", "label": "Supplier's final offer",
        "caller_keys": ["current_offer_numeric", "current_offer"]},
      "currency": {"table": "supplier_response", "column": "currency", "row_id": "id",
        "lookup": {"workflow_id": "workflow_id", "supplier_id": "supplier_id"},
        "order_by": "round_number DESC, id DESC", "value_type": "text", "label": "Currency of the offer",
        "caller_keys": ["currency", "currency_code"]},
      "supplier_contact_name": {"table": "bp_supplier", "column": "contact_name_1", "row_id": "supplier_id",
        "lookup": {"supplier_id": "supplier_id"}, "value_type": "text", "label": "Supplier contact",
        "caller_keys": ["contact_name", "supplier_contact"]}
    },
    "context": [],
    "reasoned": {},
    "never_state": {
      "target_price": ["target_price"],
      "walkaway_price": ["walkaway_price"],
      "market_floor_price": ["market_floor_price"]
    },
    "required_elements": [],
    "forbidden_patterns": {
      "bank_details": "\\b[A-Z]{2}\\d{2}[A-Z0-9]{11,30}\\b|sort\\s*code|account\\s*number|\\biban\\b|\\bswift\\b|bank\\s+details",
      "liability_admission": "we\\s+(accept|admit|acknowledge)\\s+(full\\s+)?(liability|responsibility|fault)|our\\s+fault|we\\s+are\\s+(liable|at\\s+fault)",
      "rights_waiver": "we\\s+(hereby\\s+)?waive|waive\\s+(any|our|all)\\s+(right|claim)",
      "award_commitment": "\\bwe(?:\\s+will|'ll|\\s+shall|\\s+are\\s+going\\s+to|\\s+are)\\s+(?:award(?:ing)?|plac(?:e|ing)\\s+the\\s+order|issu(?:e|ing)\\s+(?:a|the)\\s+(?:po|purchase\\s+order)|sign(?:ing)?\\s+the\\s+(?:contract|order))|\\b(?:the\\s+)?(?:contract|order|business|deal|work)\\s+(?:is|will\\s+be)\\s+yours\\b(?!\\s+to\\b)|\\byou(?:'ve|\\s+have)\\s+(?:won|got|secured)\\s+(?:the\\s+|our\\s+)?(?:contract|order|business|deal|work)|\\b(?:contract|order|business|deal)\\s+(?:will\\s+be|is\\s+being)\\s+awarded\\s+to\\s+you|\\bconsider\\s+(?:the\\s+|this\\s+)?(?:order|contract|deal)\\s+(?:placed|confirmed|done|agreed|awarded)|\\bwe(?:\\s+will|'ll)?\\s+commit\\s+to\\s+(?:order|buy|purchas|plac|a\\s+(?:minimum\\s+)?volume|the\\s+volume)|guarantee[ds]?\\s+(?:you\\s+)?(?:the\\s+|a\\s+)?(?:volume|order|business|contract)"
    }
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (
  SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailFamily_final_offer_reply');
