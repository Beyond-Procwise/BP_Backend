-- Accelerator policies: the rules a customer adopts and then edits.
-- Nothing here is compiled into the product; changing a row changes behaviour.
-- Spec: docs/superpowers/specs/2026-08-07-approvals-surface-design.md
BEGIN;

-- 1. approve_email: approving your own outbound email is a distinct, lesser
-- capability than approving money. Buyer and above hold it; transact does not
-- move. Named separately so the difference is visible in the policy, not
-- implied by rank.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(
               jsonb_set(
                   jsonb_set(
                       policy_details,
                       '{rules,roles,Buyer,allow}',
                       '["read","compute","write","approve_email"]'::jsonb, true),
                   '{rules,roles,Approver,allow}',
                   '["read","compute","write","communicate","transact","approve_email"]'::jsonb, true),
               '{rules,roles,Admin,allow}',
               '["read","compute","write","communicate","transact","share","configure","delegate","approve_email"]'::jsonb, true),
           '{rules,irreversible_classes}',
           '["communicate","transact","share","configure","delegate","approve_email"]'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition';

-- 2. Dispatch approval drops to Buyer (the drafter approves their own mail --
-- the agent drafted it on their behalf), and a draft edited after approval is
-- refused rather than sent on a stale approval.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(policy_details, '{required_role}', '"Buyer"'::jsonb, true),
           '{rules,on_content_mismatch}', '"deny"'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'email_dispatch_approval';

-- 3. The approval action itself, so authorize() has a policy to resolve.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
VALUES
(
 'EmailApprovalCapabilityPolicy', 'security',
 'Who may record an approval for an outbound email.',
 '{
   "policy_identifier": "email_approval_capability",
   "required_role": "Buyer",
   "applies_to": ["approval.email"],
   "rules": {
     "self_approval_allowed": true,
     "note": "The agent drafts on the user behalf, so approving is authorship rather than oversight. Set self_approval_allowed to false to require a second person."
   }
 }'::jsonb,
 '', 1, 1, 'accelerator_seed', now()
);

-- 4. Revoke scope: who may cancel whose approval is a policy decision too,
-- not a hardcoded rule in the router. own_or_higher_rank is the sensible
-- default -- you can undo your own decision, and anyone who strictly
-- outranks the role required to approve at all can undo someone else's.
-- any_approver keeps today's behaviour for a small team; own_only is the
-- strictest. An absent or unrecognised value falls back to
-- own_or_higher_rank, never to any_approver -- see
-- src/api/routers/approvals.py:_revoke_scope. jsonb_set (not a fresh INSERT)
-- so this lands whether item 3 above just created the row in this same run
-- or the row already existed from an earlier deploy.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,revoke_scope}', '"own_or_higher_rank"'::jsonb, true),
       version = version + 1,
       last_modified_by = 'accelerator_seed',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'email_approval_capability';

COMMIT;
