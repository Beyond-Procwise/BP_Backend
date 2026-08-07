BEGIN;

-- Removes the whole email_approval_capability row, which also reverses
-- item 4 of accelerator_policies.sql (the revoke_scope key added to this
-- same row's rules) -- there is no separate row or key left behind to clean
-- up once this DELETE has run.
DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'email_approval_capability';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(
               jsonb_set(
                   jsonb_set(
                       policy_details,
                       '{rules,roles,Buyer,allow}',
                       '["read","compute","write"]'::jsonb, true),
                   '{rules,roles,Approver,allow}',
                   '["read","compute","write","communicate","transact"]'::jsonb, true),
               '{rules,roles,Admin,allow}',
               '["read","compute","write","communicate","transact","share","configure","delegate"]'::jsonb, true),
           '{rules,irreversible_classes}',
           '["communicate","transact","share","configure","delegate"]'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details - 'rules' ||
           jsonb_build_object('rules', (policy_details->'rules') - 'on_content_mismatch'),
           '{required_role}', '"Approver"'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'email_dispatch_approval';

COMMIT;
