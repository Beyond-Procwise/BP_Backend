-- The code that reads versions must be rolled back first. Drops every edit's history.
BEGIN;
DROP TABLE IF EXISTS proc.bp_report_version;
ALTER TABLE proc.bp_report_job
    DROP COLUMN IF EXISTS last_edited_by, DROP COLUMN IF EXISTS current_version,
    DROP COLUMN IF EXISTS title, DROP COLUMN IF EXISTS fact_pack;
COMMIT;
