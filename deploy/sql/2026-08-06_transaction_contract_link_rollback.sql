-- Rollback of 2026-08-06_transaction_contract_link.sql.
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
