-- Task 4 (value ledger, 2026-09-25) needs the triage exposure_gbp for a discrepancy
-- mirror: proc.bp_triage_finding.mirror_id -> proc.bp_triage_result via finding_id,
-- ordered by proc.bp_triage_run.started_at. bp_triage_finding.mirror_id is already
-- indexed (2026-09-24_bp_triage.sql), but bp_triage_result carries no index on
-- finding_id -- only on run_id and deal_id -- so that join falls back to a full
-- Parallel Seq Scan of bp_triage_result (1.27M rows live on bp_testdb 2026-09-25) for
-- every discrepancy row the Value Found query considers (~4,100 of them), which timed
-- out GET /spendiq/value-summary outright rather than just running slow. Additive and
-- idempotent.
BEGIN;

CREATE INDEX IF NOT EXISTS ix_bp_triage_result_finding
    ON proc.bp_triage_result (finding_id);

COMMIT;
