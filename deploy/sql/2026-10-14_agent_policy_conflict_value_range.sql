-- 2026-10-14  Precedent applies only within a governed value range (conflict history and precedent, Task 12).
--
-- Adds the rule precedent_value_range_pct (default 20) to the governed agent_policy_conflicts row
-- created by 2026-10-13_agent_policy_conflict_precedent.sql, and says in the row's description
-- what the number does. Before the decision engine decides a clash on precedent, every numeric
-- condition value of the new action must be at most this many percent above the largest value
-- people approved in the cases it cites; otherwise the clash goes to people. 0 means never above
-- the largest approved value; null means no range check; a missing rule sends every clash to people.
--
-- Additive and idempotent: only an active row that does not state the rule yet is touched, so a
-- second run changes nothing and a customer's own value (or null) is never overwritten. That one
-- change is a new version of the row (version + 1), as for every other governed-limit edit.
-- Run on BOTH bp_testdb AND bp_sqldb.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,precedent_value_range_pct}', '20'::jsonb, true),
       policy_desc =
       'How many times people must have decided the same clash between two agent policies the '
       'same way, with the same policy versions, before the decision engine decides that clash '
       'itself on precedent (approve or reject, citing those cases), and the policy owners are '
       'asked to make it a standing rule. Any disagreement, or fewer decisions, sends the clash '
       'to people. 0 switches precedent and the standing-rule proposal off. '
       'precedent_value_range_pct: precedent applies only when every number the clashing '
       'policies look at (an amount, say) is at most this many percent above the largest value '
       'people approved in the cases it cites; anything higher goes to people. 20 means up to 20% '
       'above the largest approved value, 0 means not above the largest approved value, and null '
       'means no range check. There is no lower limit.',
       version = COALESCE(version, 1) + 1,
       last_modified_date = now(),
       last_modified_by = 'agent_policy_conflicts'
 WHERE policy_details->>'policy_identifier' = 'agent_policy_conflicts'
   AND policy_status = 1
   AND jsonb_typeof(policy_details->'rules') = 'object'
   AND NOT (policy_details->'rules') ? 'precedent_value_range_pct';

COMMIT;
