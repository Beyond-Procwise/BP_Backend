BEGIN;
-- The email agent's autonomy limit is governed DATA, not code: which supplier
-- replies it may answer unattended, and which must go to a human, is read from
-- this row at decision time (DecisionEngine.decide_email_reply via
-- resolve_authority). No threshold is hardcoded anywhere in the email path.
--
-- auto_reply_intents ships EMPTY on purpose. On day one every reply escalates to
-- the Action Centre; widening the list is a Policies-screen edit, not a deploy.
-- Money authority is NOT duplicated here -- defer_value_limit_to points at the
-- existing ApprovalThresholdPolicy so there is exactly one spend limit.
--
-- Keyed on (policy_type, policy_name), the natural key enforced by
-- ux_bp_policy_active_type_name (deploy/sql/2026-07-18_bp_governance_active_unique.sql).
-- policy_id is environment-specific (GENERATED ALWAYS AS IDENTITY).
-- Idempotent: re-running updates the same row rather than inserting a second.

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
VALUES (
    'EmailReplyAutonomyPolicy',
    'email_autonomy',
    'When the email agent may reply to a supplier unattended, and when it must escalate to a human.',
    jsonb_build_object(
      'policy_identifier', 'email_reply_autonomy',
      'rules', jsonb_build_object(
        'auto_reply_intents', '[]'::jsonb,
        'escalate_intents', '["price_change","terms_change","contract_variation","liability","dispute","new_commitment"]'::jsonb,
        'defer_value_limit_to', 'approval_threshold',
        'max_auto_replies_per_thread', 2,
        'min_intent_confidence', 0.8,
        'on_missing_policy', 'escalate',
        'on_ungrounded_facts', 'escalate'
      )
    ),
    'email_drafting_agent, negotiation_agent, supplier_interaction_agent',
    1,
    1,
    now(), 'implementation-plan', now(), 'implementation-plan'
)
ON CONFLICT (policy_type, policy_name) WHERE policy_status = 1
DO UPDATE SET
    policy_details     = EXCLUDED.policy_details,
    policy_linked_agents = EXCLUDED.policy_linked_agents,
    policy_desc        = EXCLUDED.policy_desc,
    version            = proc.bp_policy.version + 1,
    last_modified_date = now(),
    last_modified_by   = 'implementation-plan';

COMMIT;
