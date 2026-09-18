-- The user types the admin area assigns, mapped onto the four product roles.
--
-- The gateway's admin area (admin.roles_and_access, and the Cognito groups it
-- adds users to) names eight PROCWISE_* user types. role_assignment only knew
-- the bp-* groups, so every one of those users fell through to
-- unmapped_group_role and acted as a Viewer wherever the backend checked --
-- admins included.
--
-- The bp-* entries are kept; this only adds keys. The mapping follows what each
-- type is for: those who configure the platform (including policy) are Admin,
-- those who sign off spend are Approver, those who source and buy are Buyer.
-- A user in several groups still resolves by multiple_groups: highest_rank.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details,
           '{rules,group_to_role}',
           COALESCE(policy_details #> '{rules,group_to_role}', '{}'::jsonb)
           || '{
                "PROCWISE_TENANT_SUPER_ADMIN":        "Admin",
                "PROCWISE_ADMIN":                     "Admin",
                "PROCWISE_POLICY_MANAGER":            "Admin",
                "PROCWISE_CHIEF_PROCUREMENT_OFFICER": "Approver",
                "PROCWISE_FINANCE_REVIEWER_APPROVER": "Approver",
                "PROCWISE_CATEGORY_MANAGER":          "Buyer",
                "PROCWISE_PROCUMENT_BUYER_ANALYST":   "Buyer",
                "PROCWISE_VIEWER":                    "Viewer"
              }'::jsonb,
           true
       ),
       version = version + 1,
       last_modified_by = 'admin_area_role_mapping',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_assignment';

COMMIT;
