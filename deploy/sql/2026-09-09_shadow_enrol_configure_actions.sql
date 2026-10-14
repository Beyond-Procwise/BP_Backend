-- Enrol the five newly-gated configure/delegate actions in shadow mode.
--
-- WHY THIS SHIPS WITH THE CALL SITES, not after them.
--
-- ASK_AUTH_MODE is "off" in this environment, so require_user returns None and
-- every caller is anonymous. The gate refuses an anonymous principal any
-- irreversible action -- correctly. Which means the five call sites added
-- alongside this migration would, on their own, stop agent creation, policy
-- reload, governance reload and model training the moment they deploy.
--
-- That is the exact situation shadow mode exists for: the rule is right, and we
-- do not yet know what enforcing it costs. Enrolled, each action is evaluated,
-- the refusal is recorded, and the call proceeds. scripts/shadow_report.py then
-- answers "who was doing this, how often" from evidence rather than guesswork.
--
-- EXPIRY IS THE POINT. On 2026-10-09 these enrolments lapse and the gate starts
-- refusing for real. By then either authentication is on and roles are granted
-- (GQ0's answer was (a)), or someone has made a deliberate decision to extend.
-- What cannot happen is this quietly becoming permanent.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           policy_details, '{rules,shadow_actions}',
           '[
              {"action": "policy.reload", "until": "2026-10-09T00:00:00Z"},
              {"action": "prompt.write",  "until": "2026-10-09T00:00:00Z"},
              {"action": "agent.create",  "until": "2026-10-09T00:00:00Z"},
              {"action": "agent.delete",  "until": "2026-10-09T00:00:00Z"},
              {"action": "model.train",   "until": "2026-10-09T00:00:00Z"}
            ]'::jsonb, true
       ),
       version = version + 1,
       last_modified_by = 'shadow_enrol_configure',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'shadow_mode';

COMMIT;
