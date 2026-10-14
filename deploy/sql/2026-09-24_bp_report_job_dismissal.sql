-- 2026-09-24  Report jobs: a blocked or failed report can be dismissed
-- ---------------------------------------------------------------------------
-- A blocked or failed report job is an item in the SpendIQ Action Centre
-- (Surface = Reports) until it is dealt with: either a later run of the same
-- report and scope is released (it leaves by itself), or a person dismisses it.
-- These record who dismissed it, when and why; report.dismissed is written to
-- bp_agent_actions alongside. NULL = not dismissed.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------

BEGIN;

ALTER TABLE proc.bp_report_job
    ADD COLUMN IF NOT EXISTS dismissed_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS dismissed_by   TEXT,
    ADD COLUMN IF NOT EXISTS dismiss_reason TEXT;

COMMIT;
