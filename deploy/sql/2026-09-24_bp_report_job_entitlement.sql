-- 2026-09-24  Report jobs: the entitlement decision a job was filed under
-- ---------------------------------------------------------------------------
-- The gate's own audit row (phase=authorize, action_type=report.generate)
-- carries no trace id, so nothing links it to the report it allowed. The job
-- now keeps the decision -- action, principal, role, the policy that answered,
-- and whether shadow mode suppressed a refusal -- and the run repeats it on
-- its report.scope_resolved event. NULL on jobs filed before this column.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------

BEGIN;

ALTER TABLE proc.bp_report_job ADD COLUMN IF NOT EXISTS entitlement JSONB;

COMMIT;
