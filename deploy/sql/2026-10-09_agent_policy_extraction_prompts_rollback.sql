DELETE FROM proc.bp_prompt
 WHERE prompt_name IN ('agent_policy_extract', 'agent_policy_fix')
   AND created_by = 'deploy/sql/2026-10-09_agent_policy_extraction_prompts.sql';
