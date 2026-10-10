-- Rollback of 2026-10-17_bp_agent_policy_person.sql. Loses every remembered sign-in email: the
-- screens go back to showing the sign-in id wherever no email was stored with the row itself.
-- Never run against shared data without a user ruling.
BEGIN;
DROP TABLE IF EXISTS proc.bp_agent_policy_person;
COMMIT;
