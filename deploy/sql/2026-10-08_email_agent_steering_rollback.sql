BEGIN;
ALTER TABLE email_agent.bp_draft_capture DROP COLUMN IF EXISTS steering;
DELETE FROM proc.bp_policy WHERE policy_name = 'EmailSteeringRules' AND created_by = 'email_assurance_migration';
COMMIT;
