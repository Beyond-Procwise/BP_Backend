-- Removes exactly what 2026-10-02_contract_raw_resolved_type.sql adds.
-- This DISCARDS every recorded structure: the values are re-derivable only by
-- re-extracting each document, so take a copy first if the classifications
-- matter.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_contracts_resolved_doc_type;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS type_agreement,
    DROP COLUMN IF EXISTS resolved_role,
    DROP COLUMN IF EXISTS resolved_doc_type;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS type_agreement,
    DROP COLUMN IF EXISTS resolved_role,
    DROP COLUMN IF EXISTS resolved_doc_type;

COMMIT;
