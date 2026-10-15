-- Rollback of 2026-10-15_negotiation_advice_alternatives_bar.sql: the bar back to 93.
-- Only while it still holds the 3 that migration set.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{many_alternatives}', '93'::jsonb),
       version = COALESCE(version, 1) + 1,
       last_modified_date = now(),
       last_modified_by = 'negotiation_advice'
 WHERE policy_name = 'negotiation_advice_thresholds'
   AND (policy_details->>'many_alternatives')::numeric = 3;

COMMIT;
