-- The code that reads these columns must be rolled back first.
BEGIN;

ALTER TABLE proc.bp_report_job
    DROP COLUMN IF EXISTS dismiss_reason,
    DROP COLUMN IF EXISTS dismissed_by,
    DROP COLUMN IF EXISTS dismissed_at;

COMMIT;
