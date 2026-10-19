-- Rollback for 2026-10-19_discount_corridor_policy.sql: states the corridor and its action
-- as NOT SET again (null), so nothing is tested. payment_terms_default is left alone: it
-- was a decision of its own and bp_testdb held it before this file.
BEGIN;
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}',
         (policy_details->'rules')
         || jsonb_build_object('discount_corridor_max_pct', NULL, 'discount_corridor_action', NULL)),
       last_modified_date = now(), last_modified_by = 'sales_quote_discount_corridor'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing' AND policy_status = 1;
COMMIT;
