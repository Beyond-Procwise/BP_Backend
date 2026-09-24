-- 2026-09-24  Report jobs: a stranded job is one nobody has vouched for lately
-- ---------------------------------------------------------------------------
-- 2026-09-24_bp_report_job.sql healed a queued or running job to failed when
-- the process reading it was not the one that accepted it. That was wrong the
-- moment a second process existed: a test run, or any other server listing the
-- jobs, failed the main server's LIVE report. The worker now stamps
-- heartbeat_at on every job it holds every 30 seconds, and a job is stranded
-- only when that stamp is more than two minutes old.
--
-- Existing rows take now() and are judged from here on. Idempotent.
-- Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------

BEGIN;

ALTER TABLE proc.bp_report_job
    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now();

COMMIT;
