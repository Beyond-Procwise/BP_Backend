-- The code that reads entitlement must be rolled back first.
BEGIN;

ALTER TABLE proc.bp_report_job DROP COLUMN IF EXISTS entitlement;

COMMIT;
