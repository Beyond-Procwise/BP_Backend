-- The code that reads the page must be rolled back first. Drops every stored page.
BEGIN;
ALTER TABLE proc.bp_report_job DROP CONSTRAINT IF EXISTS ck_bp_report_job_page;
ALTER TABLE proc.bp_report_job DROP COLUMN IF EXISTS page_media_type, DROP COLUMN IF EXISTS page;
COMMIT;
