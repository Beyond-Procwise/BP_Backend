-- Removes exactly what 2026-10-04_contract_buyer_signatory.sql adds.
--
-- Discards every buyer signatory read so far. They are recoverable without
-- re-extraction -- scripts/backfill_contract_parties.py re-reads the signature
-- block from parser_snapshot->>'full_text' -- so this rollback costs nothing but
-- the column.
--
-- The COMMENT this migration added to contract_signatory_name is left in place:
-- it documents why that column has no supplier_ prefix, which stays true.
BEGIN;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS buyer_signatory_role,
    DROP COLUMN IF EXISTS buyer_signatory_name;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS buyer_signatory_role,
    DROP COLUMN IF EXISTS buyer_signatory_name;

COMMIT;
