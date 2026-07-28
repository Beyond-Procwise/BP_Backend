BEGIN;
-- Deactivate rather than delete: bp_decision rows reference policy_name/policy_id
-- and an audit trail that points at a vanished policy cannot be re-derived.
UPDATE proc.bp_policy
   SET policy_status = 0,
       last_modified_date = now(),
       last_modified_by = 'rollback-2026-07-28'
 WHERE policy_type = 'email_autonomy'
   AND policy_name = 'EmailReplyAutonomyPolicy';
COMMIT;
