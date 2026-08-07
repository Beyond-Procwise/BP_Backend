-- policies_for_action() selects on details->'applies_to'. Without it the gate
-- default-denies every irreversible action and no policy edit can open it.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{applies_to}', '["email.send"]'::jsonb, true
       ),
       version = version + 1,
       last_modified_by = 'guardrail_applies_to',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' IN (
        'email_dispatch_approval',
        'email_recipient_allowlist'
   );

COMMIT;
