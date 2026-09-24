-- The code that reads heartbeat_at must be rolled back first.
BEGIN;

ALTER TABLE proc.bp_report_job DROP COLUMN IF EXISTS heartbeat_at;

COMMIT;
