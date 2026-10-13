-- Rollback for 2026-10-13_bp_sales_quote_template.sql.
--
-- Drops the templates (edits to them included), the note on each quote of which one it
-- started from, the Admin permit and the fill prompt. No quote loses a line or a figure:
-- a template is only ever copied from. Roll the code back first -- it selects
-- template_key.

BEGIN;

DELETE FROM proc.bp_prompt
 WHERE prompt_name = 'sales_quote_template_fill'
   AND created_by = 'deploy/sql/2026-10-13_bp_sales_quote_template.sql';

UPDATE proc.bp_policy
   SET policy_status = 0, last_modified_date = now(), last_modified_by = 'sales_quote_template'
 WHERE policy_details->>'policy_identifier' = 'quote_template_write_authority'
   AND policy_status = 1;

ALTER TABLE proc.bp_sales_quote DROP COLUMN IF EXISTS template_key;

DROP TABLE IF EXISTS proc.bp_sales_quote_template;

COMMIT;
