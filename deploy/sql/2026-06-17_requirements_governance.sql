-- 2026-06-17 Governance seed for RequirementsAgent: elicitation prompt + required-fields policy.
-- Column names verified against 2026-06-08_create_bp_prompt_bp_policy.sql:
--   bp_prompt has NO `template` column — the template lives in prompts_desc (JSONB)
--   under the key `prompt_template`, which PromptEngine parses and exposes as `template`.
--   Neither prompt_name nor policy_name has a unique constraint, so seeds use the
--   INSERT ... SELECT ... WHERE NOT EXISTS idempotency pattern (not ON CONFLICT).
--   Linked-agent slug is `requirements_agent` (matches BaseAgent._governance_slug()).
BEGIN;

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT
    'requirements_elicitation',
    'elicitation',
    'requirements_agent',
    '{"prompt_template": "You are a procurement requirements assistant. Given the current requirement (JSON) and the buyer''s latest message, extract any NEW field values the message provides and ask ONE concise question for the single most important still-missing field. Never invent values not stated by the buyer. Allowed fields: title, category, description, quantity, unit, target_budget, currency, needed_by_date, delivery_location, priority. Current requirement: {requirement}. Still missing: {missing}. Buyer message: {message}. Respond ONLY with JSON: {\"updates\": {}, \"next_question\": \"\"}"}'::jsonb
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'requirements_elicitation'
);

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details, policy_linked_agents)
SELECT
    'requirement_required_fields',
    'requirements',
    'Fields that must be filled before a procurement requirement is considered complete.',
    '{"required_fields": ["title", "category", "quantity", "needed_by_date", "delivery_location"]}'::jsonb,
    'requirements_agent'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'requirement_required_fields'
);

COMMIT;
