-- Reverses 2026-09-24_bp_triage.sql. Removes the Action Centre findings triage created
-- that nobody has touched; findings a person acted on stay (their map rows go, so they
-- become ordinary findings).
BEGIN;
DELETE FROM proc.bp_detection_finding f
 USING proc.bp_triage_finding m
 WHERE m.finding_id = f.finding_id
   AND f.status = 'open' AND f.lifecycle_status = 'open'
   AND f.owner IS NULL AND f.due_date IS NULL AND f.resolved_by IS NULL;
DROP TABLE IF EXISTS proc.bp_triage_result;
DROP TABLE IF EXISTS proc.bp_triage_finding;
DROP TABLE IF EXISTS proc.bp_triage_run;
DELETE FROM proc.bp_policy
 WHERE policy_type = 'limit'
   AND policy_details->>'policy_identifier' = 'triage_tolerances';
COMMIT;
