-- P3: nobody approves their own request.
--
-- Two halves, because the rule was unenforceable without both.
--
-- 1. proc.draft_rfq_emails gains `requested_by`. The approvals surface has
--    checked since it was rebuilt that the APPROVER is the authenticated
--    principal and not a name from the body — but it never had anything to
--    compare that against, because no draft recorded who asked for it. The
--    human path in is POST /workflows/email/prepare (the report panel), which
--    now stamps the principal here. Agent-drafted rows leave it NULL, and NULL
--    is not a match: an agent-created draft has no human requester to collide
--    with, and making those unapprovable would be a broken surface rather than
--    a stricter rule.
--
-- 2. EmailApprovalCapabilityPolicy (#704) gains rules.self_approval = 'deny'.
--    The bar is a policy, not a constant, so a customer owns it like everything
--    else on this surface — and a customer who wants to permit it has to write
--    that down. A MISSING key denies (api/routers/approvals._self_approval_rule),
--    so a deployment that never runs this migration is barred rather than
--    silently unguarded. The row is what makes the bar attributable and
--    versioned, not what makes it exist.
--
-- The August migration i6_fix_remove_self_approval_allowed removed a key
-- nothing read. This adds one that something reads — the difference is the
-- point, and tests/approvals/test_self_approval_barred.py is what holds it.
BEGIN;

ALTER TABLE proc.draft_rfq_emails
  ADD COLUMN IF NOT EXISTS requested_by text;

COMMENT ON COLUMN proc.draft_rfq_emails.requested_by IS
  'Authenticated principal who prepared this draft, when a person did. NULL for '
  'agent-drafted rows. Read by the approvals surface to refuse self-approval; '
  'never accepted from a request body.';

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,self_approval}', '"deny"'::jsonb, true),
       version = version + 1,
       last_modified_by = 'self_approval_bar',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'email_approval_capability'
   AND policy_details->'rules'->>'self_approval' IS DISTINCT FROM 'deny';

COMMIT;
