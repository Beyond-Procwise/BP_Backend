-- 2026-10-03  Who may approve an imported style pack or page layout.
--
-- style_pack.approve is class `configure`, which is irreversible, so with no row here
-- guardrail.authorize never resolves and the endpoint refuses EVERYONE -- which looks like a
-- permissions bug rather than a missing row. Measured before this migration: an Admin got
-- "no policy speaks to style_pack.approve, and configure cannot be assumed safe".
-- Mirrors PlaybookAuthorityPolicy (2026-09-29) and GovernanceAuthorityPolicy, Admin-only,
-- for the same reason: approving a pack changes what every report built from it looks like.
--
-- style_pack.write (importing, renaming, defining a rating scale) is deliberately NOT here.
-- It is class `write`, which is reversible, so a Buyer may import -- and nothing imported can
-- reach a report until an Admin approves it. That split IS the gate (design section 8).
--
-- Deliberately NOT enrolled in ShadowModePolicy: this action has no callers that predate the
-- gate, so it is enforced from the first request.
--
-- Idempotent. Safe to re-run.

BEGIN;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details, policy_status,
     created_by, last_modified_by)
SELECT
    'StylePackAuthorityPolicy',
    'authority',
    'Who may approve an imported style pack or one of its page layouts. Importing one is style_pack.write and is not governed here.',
    '{"policy_identifier": "style_pack_authority",
      "applies_to": ["style_pack.approve"],
      "required_role": "Admin",
      "rules": {"effect": "allow"}}'::jsonb,
    1,
    'atb_style_pack_policy_migration_2026_10_03',
    'atb_style_pack_policy_migration_2026_10_03'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
     WHERE policy_details->>'policy_identifier' = 'style_pack_authority'
);

COMMIT;
