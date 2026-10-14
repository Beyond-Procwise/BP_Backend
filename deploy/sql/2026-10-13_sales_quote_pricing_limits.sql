-- The two pricing limits the Quote builder tests a quote against, stated as NOT SET.
--
-- The margin floor and the discount corridor are pricing-policy facts. Nobody has given
-- a value for either, and a number written here to fill the gap would be tested against
-- real quotes as if someone had decided it. So both are stated as null: "there is no
-- limit yet", which is a different statement from the rule being absent (that raises).
-- The screen shows "Not set" and the check does not run.
--
--   margin_floor_pct           percent, e.g. 30 = quotes under 30% margin are flagged.
--   discount_corridor_max_pct  percent, either one number for every category, or an
--                              object by category with an optional "default":
--                              {"Platform / Enterprise": 8, "default": 5}
--
-- To set one, update this row's rules; it is read fresh, no restart needed.
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code that reads it.

BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'SalesQuotePricingLimitPolicy', 'limit',
       'The margin floor and the discount corridor a customer quote is tested against. '
       'Both are unset (null) until pricing policy supplies a value.',
       jsonb_build_object(
         'policy_identifier', 'sales_quote_pricing',
         'rules', jsonb_build_object(
             'margin_floor_pct',          NULL,
             'discount_corridor_max_pct', NULL)),
       '', 1, 1, now(), 'sales_quote_builder', now(), 'sales_quote_builder'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'sales_quote_pricing'
       AND p.policy_status = 1);

COMMIT;
