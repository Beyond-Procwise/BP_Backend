-- Rollback for 2026-10-13_sales_quote_drafting_prompt.sql. With the row gone the agent
-- bar's endpoint answers 503 "prompt unavailable"; nothing else reads it.
BEGIN;

DELETE FROM proc.bp_prompt
 WHERE prompt_name = 'sales_quote_drafting'
   AND created_by = 'deploy/sql/2026-10-13_sales_quote_drafting_prompt.sql';

COMMIT;
