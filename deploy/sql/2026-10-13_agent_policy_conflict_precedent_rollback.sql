-- Rollback of 2026-10-13_agent_policy_conflict_precedent.sql. Removes every version of the
-- agent_policy_conflicts row, including versions a customer saved through the policy admin.
-- Afterwards precedent never applies and no standing-rule proposal is raised (both escalate /
-- skip with a warning). Never run against shared data without a user ruling.
BEGIN;
DELETE FROM proc.bp_policy WHERE policy_details->>'policy_identifier' = 'agent_policy_conflicts';
COMMIT;
