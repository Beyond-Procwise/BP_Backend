BEGIN;

UPDATE proc.bp_policy
   SET policy_details = policy_details - 'applies_to',
       version = version + 1,
       last_modified_by = 'guardrail_applies_to_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' IN (
        'email_dispatch_approval',
        'email_recipient_allowlist'
   );

COMMIT;
