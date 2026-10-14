-- 2026-09-29  Who may author and approve a playbook.
--
-- Without a row, guardrail.authorize finds no policy for these names, the
-- decision never resolves, and the endpoint refuses everyone -- which looks
-- like a permissions bug rather than a missing row. Mirrors
-- GovernanceAuthorityPolicy (policy.write / prompt.write), Admin-only.
--
-- Deliberately NOT enrolled in ShadowModePolicy. The thirteen actions there
-- were enrolled to avoid breaking callers that predated the gate; these two
-- have no callers yet, so they are enforced from the first request.
--
-- Idempotent. Safe to re-run.

BEGIN;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details, policy_status,
     created_by, last_modified_by)
SELECT
    'PlaybookAuthorityPolicy',
    'authority',
    'Who may author a playbook and who may approve one. Approving a proposal is workflow.run, not this.',
    '{"policy_identifier": "playbook_authority",
      "applies_to": ["playbook.write", "playbook.approve"],
      "required_role": "Admin",
      "rules": {"effect": "allow"}}'::jsonb,
    1,
    'bp_playbook_migration_2026_09_29',
    'bp_playbook_migration_2026_09_29'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
     WHERE policy_details->>'policy_identifier' = 'playbook_authority'
);

COMMIT;
