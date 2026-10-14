-- Enrol three detectors in the Opportunity Critic's shadow mode until 2026-10-09.
--
-- 2026-09-09_critic_governance.sql ships shadow_detectors EMPTY, by design. These
-- three were enrolled directly in bp_testdb during the critic build, with no file
-- behind them, so bp_sqldb had the critic enforcing on every detector. This file
-- records that enrolment so every database gets the same list.
--
-- Shadowed, the critic records what it would have suppressed for these detectors
-- but suppresses nothing. Every entry carries an "until".
--
-- Idempotent: only an empty list is filled; an edited list is left alone.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,shadow_detectors}',
           '[
              {"detector": "Duplicate Invoice Recovery", "until": "2026-10-09T00:00:00Z"},
              {"detector": "Invoice Overbilling",        "until": "2026-10-09T00:00:00Z"},
              {"detector": "Price Benchmark Variance",   "until": "2026-10-09T00:00:00Z"}
            ]'::jsonb, true
       ),
       version = version + 1,
       last_modified_by = 'critic_shadow_detectors',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_name = 'opportunity_critic_thresholds'
   AND coalesce(jsonb_array_length(policy_details->'rules'->'shadow_detectors'), 0) = 0;

COMMIT;
