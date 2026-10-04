-- Rollback of 2026-10-04_demand_intake_extract_prompt.sql.
--
-- Deactivates rather than deletes: prompts_status = 0 is how this table retires a prompt, and a
-- DELETE would take the audit of who added it with it. With the row inactive the extraction
-- endpoint refuses, and the intake conversation falls back to reading the text in the browser —
-- which is the behaviour that shipped before this prompt existed.
BEGIN;

UPDATE proc.bp_prompt
   SET prompts_status = 0,
       last_modified_date = now(),
       last_modified_by = 'deploy/sql/2026-10-04_demand_intake_extract_prompt_rollback.sql'
 WHERE prompt_name = 'demand_intake_extract';

COMMIT;
