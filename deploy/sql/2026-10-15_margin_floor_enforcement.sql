-- How the margin floor is enforced, as policy a user edits (Nick, 2026-10-15: "these to be
-- policy and not hard coded"). The floor itself is margin_floor_pct in the same row.
--
-- Three rules join `sales_quote_pricing`; the approval route reads them fresh, so an edit
-- applies on the next approval:
--   margin_floor_action              "refuse"   below the floor is refused, no exceptions
--                                    "override" refused unless the approver holds the
--                                               sales_quote.override_margin_floor permit
--                                    "record"   approved, and the approval records it
--   margin_floor_untested            "refuse" | "allow"  a quote with no costed line,
--                                               whose margin cannot be shown
--   margin_floor_applies_to_catalog  true | false  hold catalog (reseller) quotes to it too
-- A rule left null or given a value outside its choices reads as the strict default
-- (refuse / refuse / false).
--
-- And who may override: SalesQuoteMarginFloorOverridePolicy, required_role Admin (Nick,
-- 2026-10-15: "Admin only"). Change required_role to move it; set policy_status = 0 and
-- nobody may.
--
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb before the code that reads them.
-- Rollback: _rollback.sql beside this file.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         policy_details, '{rules}',
         jsonb_build_object('margin_floor_action', 'override',
                            'margin_floor_untested', 'refuse',
                            'margin_floor_applies_to_catalog', false)
         || (policy_details->'rules')),
       last_modified_date = now(), last_modified_by = 'sales_quote_margin_floor'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing'
   AND policy_status = 1
   AND NOT (policy_details->'rules' ?& ARRAY['margin_floor_action', 'margin_floor_untested',
                                             'margin_floor_applies_to_catalog']);

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'SalesQuoteMarginFloorOverridePolicy', 'authority',
       'Who may approve a quote whose margin is below the pricing policy''s floor, when '
       'sales_quote_pricing says margin_floor_action = override. No agent may.',
       jsonb_build_object(
         'policy_identifier', 'sales_quote_margin_floor_override',
         'applies_to', jsonb_build_array('sales_quote.override_margin_floor'),
         'required_role', 'Admin',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'sales_quote_margin_floor', now(), 'sales_quote_margin_floor'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'sales_quote_margin_floor_override');

COMMIT;
