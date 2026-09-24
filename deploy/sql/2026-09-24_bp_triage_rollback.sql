-- Reverses 2026-09-24_bp_triage.sql. Removes the Action Centre findings triage created
-- that nobody has touched; findings a person acted on stay (their map rows go, so they
-- become ordinary findings).
--
-- To undo ONE run rather than the whole feature, use scripts/triage_rollback.py
-- --run-id <uuid>. It also deletes that run's bp_triage_deal_state rows, so the scheduled
-- job will re-check those deals within its next interval unless TRIAGE_INTERVAL_MINUTES
-- is raised or the procwise service is stopped. It does not undo the supersedes or
-- in-place updates the run made to findings that already existed.
BEGIN;
-- The decision sync first, so the deletes below cannot fire it.
DROP TRIGGER IF EXISTS tr_bp_triage_finding_decision_to_mirror ON proc.bp_detection_finding;
DROP TRIGGER IF EXISTS tr_bp_triage_mirror_decision_to_finding ON proc.bp_extraction_discrepancy;
DROP FUNCTION IF EXISTS proc.bp_triage_finding_decision_to_mirror();
DROP FUNCTION IF EXISTS proc.bp_triage_mirror_decision_to_finding();
DELETE FROM proc.bp_detection_finding f
 USING proc.bp_triage_finding m
 WHERE m.finding_id = f.finding_id
   AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL;
-- Their Action Centre mirror rows go the same way: only those nobody has touched.
DELETE FROM proc.bp_extraction_discrepancy WHERE source_file LIKE 'triage:%' AND status='open'
   AND resolved_by IS NULL AND query_sent_at IS NULL;
DROP TABLE IF EXISTS proc.bp_triage_deal_state;
DROP TABLE IF EXISTS proc.bp_triage_result;
DROP TABLE IF EXISTS proc.bp_triage_finding;
DROP TABLE IF EXISTS proc.bp_triage_run;
DELETE FROM proc.bp_policy
 WHERE policy_type = 'limit'
   AND policy_details->>'policy_identifier' = 'triage_tolerances';
COMMIT;
