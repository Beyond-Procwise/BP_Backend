-- Rollback of 2026-10-16-email-settings.sql.
-- WARNING: drops proc.bp_internal_email, which drops any pending internal emails, and
-- deletes the email_policy row with its history.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_internal_email_author_status;
DROP TABLE IF EXISTS proc.bp_internal_email;
DELETE FROM proc.bp_policy WHERE policy_name = 'email_policy' AND policy_type = 'setting';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         jsonb_set(policy_details #- '{rules,roles,SecurityReviewer}',
                   '{rules,roles,Admin,allow}',
                   COALESCE((SELECT jsonb_agg(e) FROM jsonb_array_elements(policy_details #> '{rules,roles,Admin,allow}') e
                              WHERE e <> '"inspect"'::jsonb), '[]'::jsonb), true),
         '{rules,reversible_classes}',
         COALESCE((SELECT jsonb_agg(e) FROM jsonb_array_elements(COALESCE(policy_details #> '{rules,reversible_classes}', '[]'::jsonb)) e
                    WHERE e <> '"inspect"'::jsonb), '[]'::jsonb), true),
       version = version + 1,
       last_modified_by = 'rollback:2026-10-16-email-settings',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

UPDATE proc.bp_policy
   SET policy_details = policy_details #- '{rules,group_to_role,PROCWISE_SECURITY_REVIEWER}',
       version = version + 1,
       last_modified_by = 'rollback:2026-10-16-email-settings',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_assignment'
   AND policy_details #> '{rules,group_to_role}' ? 'PROCWISE_SECURITY_REVIEWER';
COMMIT;
