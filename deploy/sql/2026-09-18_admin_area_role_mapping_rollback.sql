-- Removing the mapping sends every admin-area user back to Viewer; it grants nothing.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = policy_details
           #- '{rules,group_to_role,PROCWISE_TENANT_SUPER_ADMIN}'
           #- '{rules,group_to_role,PROCWISE_ADMIN}'
           #- '{rules,group_to_role,PROCWISE_POLICY_MANAGER}'
           #- '{rules,group_to_role,PROCWISE_CHIEF_PROCUREMENT_OFFICER}'
           #- '{rules,group_to_role,PROCWISE_FINANCE_REVIEWER_APPROVER}'
           #- '{rules,group_to_role,PROCWISE_CATEGORY_MANAGER}'
           #- '{rules,group_to_role,PROCWISE_PROCUMENT_BUYER_ANALYST}'
           #- '{rules,group_to_role,PROCWISE_VIEWER}',
       version = version + 1,
       last_modified_by = 'admin_area_role_mapping_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_assignment';

COMMIT;
