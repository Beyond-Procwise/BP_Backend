BEGIN;
UPDATE proc.bp_policy
   SET policy_details = policy_details #- '{rules,group_to_role,procwise_Admin}' #- '{rules,group_to_role,procwise_viewer}',
       version = version + 1,
       last_modified_by = 'rollback:2026-10-19-group-to-role-observed-spellings',
       last_modified_date = now()
 WHERE policy_status = 1 AND policy_details->>'policy_identifier' = 'role_assignment';
COMMIT;
