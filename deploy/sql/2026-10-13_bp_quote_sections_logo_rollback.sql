-- Rollback for 2026-10-13_bp_quote_sections_logo.sql. Roll the code back first.
--
-- Drops every uploaded logo and every quote's own layout (headings, section order and
-- the words of free-text sections; scope, assumptions, lines and terms are untouched).
-- The sum prompt goes with it.

BEGIN;

DELETE FROM proc.bp_prompt
 WHERE prompt_name = 'sales_quote_sum'
   AND created_by = 'deploy/sql/2026-10-13_bp_quote_sections_logo.sql';


ALTER TABLE proc.bp_sales_quote DROP COLUMN IF EXISTS sections;
DROP TABLE IF EXISTS proc.bp_brand_logo;

COMMIT;
