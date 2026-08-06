-- Rollback of 2026-08-06_transaction_contract_link.sql.
--
-- ORDERING — REVERT THE YAML FIRST, THEN RUN THIS.
-- extraction_schemas/{invoice,purchase_order,quote}.yaml declare contract_id as
-- a db_column. load_all_schemas() checks every declared column against
-- information_schema on EVERY schema load, and it runs in the API lifespan
-- (src/api/main.py). Running this script while commit e50cba9's YAML is still
-- deployed raises SchemaDriftError and the API will not start.
--
-- Correct order: revert/withdraw the YAML commit, deploy that, then run this.
-- (Forward order is the mirror image: migration first, YAML second.)
--
-- Drops the columns this migration added. It does NOT drop
-- ix_bp_purchase_order_trgt_contract_id's column — that column predates this
-- migration — only the index the migration created on it.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_quote_trgt_contract_id;
DROP INDEX IF EXISTS proc.ix_bp_invoice_trgt_contract_id;
DROP INDEX IF EXISTS proc.ix_bp_purchase_order_trgt_contract_id;

ALTER TABLE proc.bp_quote_raw    DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_quote_stg    DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_quote_trgt   DROP COLUMN IF EXISTS contract_id;

ALTER TABLE proc.bp_invoice_raw  DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_invoice_stg  DROP COLUMN IF EXISTS contract_id;
ALTER TABLE proc.bp_invoice_trgt DROP COLUMN IF EXISTS contract_id;

COMMIT;
