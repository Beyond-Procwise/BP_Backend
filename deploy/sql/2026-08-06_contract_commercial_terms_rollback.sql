-- Rollback of 2026-08-06_contract_commercial_terms.sql.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_contracts_parent_contract_id;

ALTER TABLE proc.bp_contract_raw
    DROP COLUMN IF EXISTS term_months,
    DROP COLUMN IF EXISTS billing_frequency,
    DROP COLUMN IF EXISTS escalator_pct,
    DROP COLUMN IF EXISTS escalator_basis,
    DROP COLUMN IF EXISTS escalator_cap_pct,
    DROP COLUMN IF EXISTS amendment_ref,
    DROP COLUMN IF EXISTS document_version;

ALTER TABLE proc.bp_contracts
    DROP COLUMN IF EXISTS term_months,
    DROP COLUMN IF EXISTS billing_frequency,
    DROP COLUMN IF EXISTS escalator_pct,
    DROP COLUMN IF EXISTS escalator_basis,
    DROP COLUMN IF EXISTS escalator_cap_pct,
    DROP COLUMN IF EXISTS amendment_ref,
    DROP COLUMN IF EXISTS document_version;

COMMIT;
