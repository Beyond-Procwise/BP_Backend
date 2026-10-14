-- Rollback of 2026-10-16_bp_agent_policy_actor_names.sql. Loses the recorded email of every second
-- review and reversal (the sign-in ids stay). Never run against shared data without a user ruling.
BEGIN;
ALTER TABLE proc.bp_policy_firing_reversal DROP COLUMN IF EXISTS reversed_by_email;
ALTER TABLE proc.bp_agent_policy_second_review DROP COLUMN IF EXISTS reviewed_by_email;
COMMIT;
