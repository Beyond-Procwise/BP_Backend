BEGIN;
-- Deactivate rather than delete: bp_decision rows reference policy_name/policy_id
-- and an audit trail that points at a vanished policy cannot be re-derived.
--
-- policy_status = 1 in the WHERE clause on purpose: history accumulates (this
-- is the same table the migration's own comment describes as version history
-- for superseded rows), so without this filter a rollback would re-stamp
-- last_modified_date/last_modified_by onto already-inactive historical rows
-- too, overwriting who and when THEY were actually last touched. Only the
-- currently-active row is this rollback's business.
UPDATE proc.bp_policy
   SET policy_status = 0,
       last_modified_date = now(),
       last_modified_by = 'rollback-2026-07-28'
 WHERE policy_type = 'email_autonomy'
   AND policy_name = 'EmailReplyAutonomyPolicy'
   AND policy_status = 1;
COMMIT;
