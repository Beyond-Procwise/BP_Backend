-- Workaround (option 2): the sign-in carries groups spelled `procwise_Admin` / `procwise_viewer`,
-- but group_to_role matches case-sensitively and only knows the upper-case spellings, so an
-- Admin was read as Viewer. Map the observed spellings. The lasting fix is a case-insensitive
-- lookup in services/rbac.resolve_roles.
BEGIN;
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,group_to_role}',
           COALESCE(policy_details #> '{rules,group_to_role}', '{}'::jsonb)
           || '{"procwise_Admin":"Admin","procwise_viewer":"Viewer"}'::jsonb, true),
       version = version + 1,
       last_modified_by = 'migration:2026-10-19-group-to-role-observed-spellings',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_assignment'
   AND NOT (policy_details #> '{rules,group_to_role}' ? 'procwise_Admin'
            AND policy_details #> '{rules,group_to_role}' ? 'procwise_viewer');
COMMIT;
