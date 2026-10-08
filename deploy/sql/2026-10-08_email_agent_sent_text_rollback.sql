-- Removes the sent-text table (and the raw text in it), the text_expired_at column and the retention policy row.
BEGIN;
DROP TABLE IF EXISTS email_agent.bp_draft_sent_text;
ALTER TABLE email_agent.bp_draft_capture DROP COLUMN IF EXISTS text_expired_at;
DELETE FROM proc.bp_policy WHERE policy_name = 'EmailTextRetention' AND created_by = 'email_assurance_migration';
COMMIT;
