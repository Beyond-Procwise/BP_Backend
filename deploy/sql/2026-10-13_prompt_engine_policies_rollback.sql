-- Rollback of 2026-10-13_prompt_engine_policies.sql.
-- Rows are retired, not deleted, as every other policy rollback does.
UPDATE proc.bp_policy
   SET policy_status = 0, last_modified_by = 'prompt_engine_rollback', last_modified_date = now()
 WHERE policy_status = 1
   AND policy_name IN ('PromptEnginePolicy', 'PromptWordingAuthorityPolicy',
                       'PromptAuthorAuthorityPolicy', 'PromptPromoteAuthorityPolicy');
