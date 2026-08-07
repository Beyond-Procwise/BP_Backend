BEGIN;

UPDATE proc.bp_policy
   SET policy_details = policy_details #- '{rules,reversible_classes}',
       version = version + 1,
       last_modified_by = 'guardrail_reversible_classes_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

COMMIT;
