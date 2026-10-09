-- Rollback of 2026-10-14_agent_policy_conflict_value_range.sql. Removes precedent_value_range_pct
-- from the agent_policy_conflicts row, INCLUDING a value a customer set through the policy admin,
-- and restores the 2026-10-13 description. Afterwards every clash that would be decided on
-- precedent goes to people instead ("precedent value range unavailable"). Never run against
-- shared data without a user ruling.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}', (policy_details->'rules') - 'precedent_value_range_pct'),
       policy_desc =
       'How many times people must have decided the same clash between two agent policies the '
       'same way, with the same policy versions, before the decision engine decides that clash '
       'itself on precedent (approve or reject, citing those cases), and the policy owners are '
       'asked to make it a standing rule. Any disagreement, or fewer decisions, sends the clash '
       'to people. 0 switches precedent and the standing-rule proposal off.',
       version = COALESCE(version, 1) + 1,
       last_modified_date = now(),
       last_modified_by = 'agent_policy_conflicts'
 WHERE policy_details->>'policy_identifier' = 'agent_policy_conflicts'
   AND (policy_details->'rules') ? 'precedent_value_range_pct';

COMMIT;
