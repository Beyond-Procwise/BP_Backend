-- Undoes QDA B6. Only before any real quote has been issued with a number: dropping the
-- column forgets every number given out, and dropping the counter restarts the series, so
-- the next quote issued would reuse a number a customer already holds. The server must be
-- back on code that does not read quote_number first.
--
-- tenant_id is left as 'default': nothing reads a NULL there as meaning anything.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_sales_quote_quote_number;
ALTER TABLE proc.bp_sales_quote DROP COLUMN IF EXISTS quote_number;
DROP TABLE IF EXISTS proc.bp_doc_number_counter;

COMMIT;
