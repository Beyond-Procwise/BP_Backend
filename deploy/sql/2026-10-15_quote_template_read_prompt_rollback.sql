-- Rollback for 2026-10-15_quote_template_read_prompt.sql. Roll the code back first:
-- without the row, POST /spendiq/quote-templates/read answers 503.

BEGIN;

DELETE FROM proc.bp_prompt
 WHERE prompt_name = 'sales_quote_template_read'
   AND created_by = 'deploy/sql/2026-10-15_quote_template_read_prompt.sql';

COMMIT;
