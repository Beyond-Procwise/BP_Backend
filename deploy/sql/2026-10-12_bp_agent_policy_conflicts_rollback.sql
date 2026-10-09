-- Rollback for 2026-10-12_bp_agent_policy_conflicts.sql.
-- Never run against shared data without a user ruling.
BEGIN;
ALTER TABLE proc.bp_policy_notification DROP CONSTRAINT IF EXISTS ck_bp_policy_notification_target;
DELETE FROM proc.bp_policy_notification WHERE firing_id IS NULL AND link LIKE 'conflict:%';
ALTER TABLE proc.bp_policy_notification ALTER COLUMN firing_id SET NOT NULL;
DROP TABLE IF EXISTS proc.bp_agent_policy_conflict_rule;
DROP TABLE IF EXISTS proc.bp_agent_policy_conflict;
COMMIT;
