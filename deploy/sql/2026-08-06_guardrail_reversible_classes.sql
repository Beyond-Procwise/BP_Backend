BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details,
           '{rules,reversible_classes}',
           '["read","compute","write"]'::jsonb,
           true
       ),
       version = version + 1,
       last_modified_by = 'guardrail_reversible_classes',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

COMMIT;
