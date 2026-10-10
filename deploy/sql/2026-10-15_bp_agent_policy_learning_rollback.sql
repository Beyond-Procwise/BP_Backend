-- Rollback of 2026-10-15_bp_agent_policy_learning.sql. Loses every suggestion, second review and
-- reversal record. Never run against shared data without a user ruling.
BEGIN;
DROP TABLE IF EXISTS proc.bp_policy_firing_reversal;
DROP FUNCTION IF EXISTS proc.bp_policy_firing_reversal_guard();
DROP TABLE IF EXISTS proc.bp_agent_policy_second_review;
DROP TABLE IF EXISTS proc.bp_agent_policy_suggestion;
COMMIT;
