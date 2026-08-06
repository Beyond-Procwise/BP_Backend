-- 2026-08-06 Phase 1a: the contract link on quote and invoice documents.
--
-- purchase_order already has contract_id on _raw/_stg/_trgt (all text, all NULL,
-- because no extraction schema declared it). Quote and invoice have no such
-- column at any layer, so a quote that cites its governing agreement has
-- nowhere to record it.
--
-- Additive and idempotent. Type is `text` to match bp_purchase_order_*.contract_id
-- and bp_contracts.contract_id, so a future join needs no cast.
BEGIN;

ALTER TABLE proc.bp_quote_raw    ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_quote_stg    ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_quote_trgt   ADD COLUMN IF NOT EXISTS contract_id TEXT;

ALTER TABLE proc.bp_invoice_raw  ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_invoice_stg  ADD COLUMN IF NOT EXISTS contract_id TEXT;
ALTER TABLE proc.bp_invoice_trgt ADD COLUMN IF NOT EXISTS contract_id TEXT;

-- Indexed because every Phase 3.2 baseline-integrity check joins a transaction
-- to its governing contract on this column.
CREATE INDEX IF NOT EXISTS ix_bp_quote_trgt_contract_id
    ON proc.bp_quote_trgt (contract_id);
CREATE INDEX IF NOT EXISTS ix_bp_invoice_trgt_contract_id
    ON proc.bp_invoice_trgt (contract_id);
CREATE INDEX IF NOT EXISTS ix_bp_purchase_order_trgt_contract_id
    ON proc.bp_purchase_order_trgt (contract_id);

COMMIT;
