-- Rollback of 2026-10-04_bp_report_job_snapshot.sql.
-- Drops the snapshots. The deck and the page are untouched, and a report can be run again.
BEGIN;
ALTER TABLE proc.bp_report_job DROP COLUMN IF EXISTS snapshot;
ALTER TABLE proc.bp_report_job DROP COLUMN IF EXISTS pack_key;
COMMIT;
