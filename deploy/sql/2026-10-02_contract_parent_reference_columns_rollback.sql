-- Removes exactly what 2026-10-02_contract_parent_reference_columns.sql adds.
-- Discards every extracted reference; they are recoverable only by re-extraction.
BEGIN;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS parent_agreement_ref,
    DROP COLUMN IF EXISTS framework_ref;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS parent_agreement_ref,
    DROP COLUMN IF EXISTS framework_ref;

COMMIT;
