-- email_agent: what steered each draft, and the switch + limits for steering.
--
-- NOT applied anywhere. Belongs to PACK (b) with the tone rules and prompts (ruling 2026-10-08: held back until live
-- verification), because steering changes what the model is asked to write and its effect on quality cannot be
-- judged without a real model.
--
-- Steering = tone directives (only for tone that came from data or the person's words), the draft author's own
-- approved style rules, and approved exemplars. The column records WHICH, by id, never the text.

BEGIN;

ALTER TABLE email_agent.bp_draft_capture ADD COLUMN IF NOT EXISTS steering JSONB;
COMMENT ON COLUMN email_agent.bp_draft_capture.steering IS
    'What steered the draft: {status, tone:[{variable,value}], style_rule_ids, exemplar_ids, exemplar_scope}. NULL = steering never ran for this row.';

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT 'EmailSteeringRules', 'email_steering',
 'Switch and limits for steering email drafts with tone, the author''s approved style rules and approved exemplars.',
 $json${
  "policy_identifier": "email_steering_rules",
  "required_role": "Admin",
  "rules": {
    "enabled": true,
    "max_style_rules": 5,
    "max_exemplars": 2,
    "max_exemplar_chars": 1200
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailSteeringRules');

COMMIT;
