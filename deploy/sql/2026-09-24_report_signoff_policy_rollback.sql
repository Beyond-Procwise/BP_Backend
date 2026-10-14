-- Governance rows are deactivated, never deleted. With both inactive, report.signoff
-- has no permit (every sign-off refused) and every report is held for sign-off.
BEGIN;

UPDATE proc.bp_policy SET policy_status = 0, last_modified_date = now(),
       last_modified_by = 'report_signoff_rollback'
 WHERE policy_details->>'policy_identifier' IN ('report_signoff', 'report_signoff_authority')
   AND policy_status = 1;

COMMIT;
