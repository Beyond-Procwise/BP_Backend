-- Rollback of 2026-08-06_contract_commercial_terms.sql.
--
-- ORDERING — REVERT THE YAML FIRST, THEN RUN THIS.
-- extraction_schemas/contract.yaml declares term_months, billing_frequency,
-- escalator_pct, escalator_basis, escalator_cap_pct, amendment_ref and
-- document_version as db_columns. load_all_schemas() checks every declared
-- column against information_schema on EVERY schema load, and it runs in the
-- API lifespan (src/api/main.py). Running this script while commits 3eff78a /
-- d825111's YAML is still deployed raises SchemaDriftError and the API will
-- not start.
--
-- Correct order: revert/withdraw the YAML commits, deploy that, then run this.
-- (Forward order is the mirror image: migration first, YAML second.)
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
