-- 2026-09-24  Index report sign-off lookups on proc.bp_approval.
-- A report job's sign-off decisions are found by grounding->>'report_job_id' (newest wins);
-- the Reports screen and Action Centre read them for every listed job, in one query.
-- Not partial: a "grounding ? 'report_job_id'" predicate is not implied by the lookup's
-- "grounding->>'report_job_id' = ANY(...)", so the planner would never use it (checked).
-- Rows without a report job id index as NULL. Idempotent. Run against: bp_testdb, bp_sqldb.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_approval_report_job;
CREATE INDEX ix_bp_approval_report_job
    ON proc.bp_approval ((grounding->>'report_job_id'), created_date DESC, approval_id DESC);
COMMIT;
