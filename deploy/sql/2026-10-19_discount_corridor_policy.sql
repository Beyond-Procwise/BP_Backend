-- The discount corridor, stated and enforced as policy (Nick, 2026-10-19: "15%, and make
-- it part of a policy and configuration decision").
--
-- In `sales_quote_pricing`:
--   discount_corridor_max_pct  15   the deepest discount a seller may type on a line.
--                                   A number, or {"<category>": n, "default": n}.
--   discount_corridor_action   "override"   a quote outside it is refused unless the
--       approver holds the sales_quote.override_margin_floor permit (Admin); "refuse"
--       allows no exception; "record" approves and records it. Same three choices, and
--       the same permit, as margin_floor_action.
-- Also states payment_terms_default on a database that has none (Nick, 2026-10-19: apply
-- to bp_sqldb); bp_testdb already has it.
--
-- Idempotent: a rule already holding a value keeps it, so an edit made in the policy
-- admin is never overwritten by a re-run. Apply to BOTH bp_testdb and bp_sqldb, before
-- the code that reads them. Rollback: _rollback.sql beside this file.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}',
         (policy_details->'rules')
         || CASE WHEN policy_details->'rules'->'discount_corridor_max_pct' IS NULL
                   OR jsonb_typeof(policy_details->'rules'->'discount_corridor_max_pct') = 'null'
                 THEN jsonb_build_object('discount_corridor_max_pct', 15)
                 ELSE '{}'::jsonb END
         || CASE WHEN policy_details->'rules'->'discount_corridor_action' IS NULL
                   OR jsonb_typeof(policy_details->'rules'->'discount_corridor_action') = 'null'
                 THEN jsonb_build_object('discount_corridor_action', 'override')
                 ELSE '{}'::jsonb END
         || CASE WHEN policy_details->'rules'->'payment_terms_default' IS NULL
                   OR jsonb_typeof(policy_details->'rules'->'payment_terms_default') = 'null'
                 THEN jsonb_build_object('payment_terms_default', '30 days from invoice')
                 ELSE '{}'::jsonb END),
       last_modified_date = now(), last_modified_by = 'sales_quote_discount_corridor'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing'
   AND policy_status = 1;

COMMIT;
