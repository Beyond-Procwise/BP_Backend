-- Prompt engine: the policy rows it is governed by.
-- Design and decision log: specs/2026-10-13-prompt-engine-phase1-design.md
--
-- 1. PromptEnginePolicy (a `limit` row, read through services/governed_limits):
--      manifest_retention_days   7     how long rendered prompt text is kept
--                                      (owner ruling 2026-10-13).
--      wording_read_passthrough  true  whether a prompt's own wording is passed
--                                      through the output-safety scrubber to the
--                                      prompt screens (owner ruling 2026-10-13:
--                                      fine, provided every read is logged, it is
--                                      internal, and it is a policy decision).
--                                      Set false and the wording is scrubbed again.
--
-- 2. Three authority rows, in the canonical action vocabulary:
--      prompt.read      Buyer     see a prompt's wording (every attempt is logged)
--      prompt.author    Buyer     save a new version (changes nothing in use)
--      prompt.promote   Approver  promote or roll back (and not the author, which
--                                 the registry enforces)
--
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code that reads them.
-- Rollback: 2026-10-13_prompt_engine_policies_rollback.sql

BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'PromptEnginePolicy', 'limit',
       'How long rendered prompt text is kept, and whether a prompt''s own wording '
       'is shown unaltered on the prompt screens.',
       jsonb_build_object(
         'policy_identifier', 'prompt_engine',
         'rules', jsonb_build_object(
             'manifest_retention_days',  7,
             'wording_read_passthrough', true)),
       '', 1, 1, now(), 'prompt_engine', now(), 'prompt_engine'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'prompt_engine'
       AND p.policy_status = 1);

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT v.name, 'authority', v.descr,
       jsonb_build_object(
         'policy_identifier', v.slug,
         'required_role', v.role,
         'applies_to', v.actions,
         'rules', jsonb_build_object('effect', 'allow')
       ),
       '', 1, 1, now(), 'prompt_engine', now(), 'prompt_engine'
  FROM (VALUES
    ('PromptWordingAuthorityPolicy', 'prompt_wording_authority',
     'Who may see a prompt''s wording. Internal; every attempt is recorded.',
     'Buyer',    '["prompt.read"]'::jsonb),
    ('PromptAuthorAuthorityPolicy',  'prompt_author_authority',
     'Who may save a new version of a prompt. Saving changes nothing in use.',
     'Buyer',    '["prompt.author"]'::jsonb),
    ('PromptPromoteAuthorityPolicy', 'prompt_promote_authority',
     'Who may put a prompt version into use or roll one back. Never its author.',
     'Approver', '["prompt.promote"]'::jsonb)
  ) AS v(name, slug, descr, role, actions)
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_name = v.name AND p.policy_status = 1);

COMMIT;
