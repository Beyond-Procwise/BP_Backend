-- deploy/sql/2026-07-28_draft_rfq_emails_attachments_rollback.sql
BEGIN;
ALTER TABLE proc.draft_rfq_emails DROP COLUMN IF EXISTS attachments;
COMMIT;
