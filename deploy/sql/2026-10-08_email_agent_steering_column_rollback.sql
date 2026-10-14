BEGIN;
ALTER TABLE email_agent.bp_draft_capture DROP COLUMN IF EXISTS steering;
COMMIT;
