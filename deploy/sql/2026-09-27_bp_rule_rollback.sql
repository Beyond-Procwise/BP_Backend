-- Rollback for 2026-09-27_bp_rule.sql.
--
-- Reactivates the five opportunity policy rows and drops the rule table.
-- Run this only alongside reverting the code: the miner reads proc.bp_rule and
-- refuses to detect anything without it, so dropping the table while the new
-- code is deployed stops opportunity mining (loudly, by design).

BEGIN;

UPDATE proc.bp_policy
   SET policy_status      = 1,
       last_modified_date = now(),
       last_modified_by   = 'bp_rule_rollback_2026_09_27'
 WHERE policy_type = 'opportunity'
   AND policy_status = 0
   AND last_modified_by = 'bp_rule_migration_2026_09_27';

DROP TABLE IF EXISTS proc.bp_rule;

COMMIT;
