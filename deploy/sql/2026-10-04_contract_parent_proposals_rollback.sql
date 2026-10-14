-- Rollback: remove the two contract parent-proposal limits.
--
-- After this, governed_limits.limit() RAISES for both keys, which means the
-- promotion hook proposes nothing (it fails closed) and the backstop sweep is
-- not scheduled. That is a full stand-down, not a half-on state.
--
-- It does NOT remove proposals already written. They are rows in
-- proc.bp_extraction_discrepancy that a person may be working; deleting
-- somebody's queue is not a rollback. To clear them deliberately:
--   DELETE FROM proc.bp_extraction_discrepancy
--    WHERE issue_type = 'contract_parent_proposed'
--      AND coalesce(status,'open') <> 'resolved';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         policy_details,
         '{rules}',
         (policy_details -> 'rules')
           - 'contract_parent_proposals_enabled'
           - 'contract_parent_proposal_sweep_hours'),
       last_modified_date = NOW(),
       last_modified_by   = 'deploy/sql/2026-10-04_contract_parent_proposals_rollback.sql',
       version            = COALESCE(version, 1) + 1
 WHERE policy_details ->> 'policy_identifier' = 'autonomous_operation';

SELECT policy_name, jsonb_object_keys(policy_details -> 'rules') AS every_rule
  FROM proc.bp_policy
 WHERE policy_details ->> 'policy_identifier' = 'autonomous_operation';
