-- A permit is stated, not inferred from the absence of a denial.
--
-- guardrail._evaluate used to treat any policy that matched an action and did
-- not deny it as the policy PERMITTING that action. Matching is applicability;
-- it is not agreement. The consequence was demonstrable: a supplier-ranking
-- weights table, given an applies_to, authorised report export.
--
-- The gate now requires rules.effect = 'allow' to permit, and treats a matching
-- policy that states no effect as UNRESOLVED -- a question for a person rather
-- than a silent grant.
--
-- These three rows are the ones that genuinely do permit today, so they now say
-- so. Without this the send path defers instead of sending.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,effect}', '"allow"'::jsonb, true
       ),
       version = version + 1,
       last_modified_by = 'stated_permits',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' IN (
        'email_dispatch_approval',      -- #673, permits email.send
        'email_recipient_allowlist',    -- #674, permits email.send
        'email_approval_capability'     -- #704, permits approval.email
   );

COMMIT;
