-- Rollback for 2026-07-30_requirements_scope_proposal_prompt.sql.
-- Removes the scoping prompt and restores the 2026-06-17 elicitation text verbatim.
-- Note: with the scoping prompt gone the agent still proposes a scope, using the
-- code default in src/agents/requirements_agent.py (_DEFAULT_SCOPE_PROMPT). To
-- disable proposing entirely, revert the agent code as well.
BEGIN;

DELETE FROM proc.bp_prompt WHERE prompt_name = 'requirements_scope_proposal';

UPDATE proc.bp_prompt
SET prompts_desc = '{"prompt_template": "You are a procurement requirements assistant. Given the current requirement (JSON) and the buyer''s latest message, extract any NEW field values the message provides and ask ONE concise question for the single most important still-missing field. Never invent values not stated by the buyer. Allowed fields: title, category, description, quantity, unit, target_budget, currency, needed_by_date, delivery_location, priority. Current requirement: {requirement}. Still missing: {missing}. Buyer message: {message}. Respond ONLY with JSON: {\"updates\": {}, \"next_question\": \"\"}"}'::jsonb,
    version = COALESCE(version, 2) - 1,
    last_modified_date = now(),
    last_modified_by = 'deploy/sql/2026-07-30_requirements_scope_proposal_prompt_rollback.sql'
WHERE prompt_name = 'requirements_elicitation';

COMMIT;
