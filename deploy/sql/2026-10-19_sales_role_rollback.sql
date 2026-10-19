-- Removing the role sends a bp-sales user back to unmapped_group_role (Viewer); it grants nothing.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = policy_details #- '{rules,roles,Sales}',
       version = version + 1, last_modified_by = 'sales_role_rollback', last_modified_date = now()
 WHERE policy_status = 1 AND policy_details->>'policy_identifier' = 'role_definition'
   AND (policy_details #> '{rules,roles}') ? 'Sales';

UPDATE proc.bp_policy
   SET policy_details = policy_details #- '{rules,group_to_role,bp-sales}',
       version = version + 1, last_modified_by = 'sales_role_rollback', last_modified_date = now()
 WHERE policy_status = 1 AND policy_details->>'policy_identifier' = 'role_assignment'
   AND (policy_details #> '{rules,group_to_role}') ? 'bp-sales';

COMMIT;
