BEGIN;
DROP TABLE IF EXISTS email_agent.bp_inbound_auth;
DELETE FROM proc.bp_policy WHERE policy_name = 'EmailSenderAuthRules' AND created_by = 'email_assurance_migration';
COMMIT;
