-- 2026-10-13  Conflict history and precedent (specs/2026-10-09-conflict-history-and-precedent-design.md §3.3).
--
-- How many times people must have decided the same clash between agent policies the same way
-- before the decision engine decides it on precedent becomes a governed limit, so a customer can
-- change it (user ruling R4). One row, the established pattern of the 2026-09-10 governed limits:
-- policy_type 'limit', DELIBERATELY NO applies_to (configuration read by name, never an
-- authority statement). The value is copied from the company setting it replaces
-- (bp_admin_config.agent_policy_settings.live_conflict_repeat), else the ruled default 5.
-- Additive and idempotent: nothing is inserted while an active row exists.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'AgentPolicyConflictPolicy', 'limit',
       'How many times people must have decided the same clash between two agent policies the '
       'same way, with the same policy versions, before the decision engine decides that clash '
       'itself on precedent (approve or reject, citing those cases), and the policy owners are '
       'asked to make it a standing rule. Any disagreement, or fewer decisions, sends the clash '
       'to people. 0 switches precedent and the standing-rule proposal off.',
       jsonb_build_object('policy_identifier', 'agent_policy_conflicts',
                          'rules', jsonb_build_object('precedent_count', n.value)),
       '', 1, 1, now(), 'agent_policy_conflicts', now(), 'agent_policy_conflicts'
  FROM (SELECT COALESCE(
          (SELECT (c.config_value->>'live_conflict_repeat')::int
             FROM proc.bp_admin_config c
            WHERE c.config_key = 'agent_policy_settings'
              AND c.config_value->>'live_conflict_repeat' ~ '^[0-9]{1,6}$'),
          5) AS value) AS n
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'agent_policy_conflicts'
       AND p.policy_status = 1
 );

COMMIT;
