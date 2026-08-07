-- Rollback for 2026-08-07_opportunity_structured_columns.sql
--
-- ORDERING WARNING. Drop the index and the CHECK constraint before the columns
-- they depend on. Dropping a column first takes its constraint with it
-- implicitly, which succeeds but leaves the DROP CONSTRAINT below to fail on a
-- re-run -- so the rollback stops being idempotent exactly when it is most
-- likely to be re-run.
--
-- Destructive: this discards the harvested structured values. They can be
-- rebuilt by re-running scripts/backfill_opportunity_structured.py, because
-- calculation_details is still written.
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_opportunity_facts_state;

ALTER TABLE proc.bp_opportunity
    DROP CONSTRAINT IF EXISTS ck_bp_opportunity_facts_state;

ALTER TABLE proc.bp_opportunity
    DROP COLUMN IF EXISTS currency,
    DROP COLUMN IF EXISTS amount_native,
    DROP COLUMN IF EXISTS unit_price,
    DROP COLUMN IF EXISTS quantity,
    DROP COLUMN IF EXISTS uom,
    DROP COLUMN IF EXISTS uom_normalised,
    DROP COLUMN IF EXISTS fx_rate,
    DROP COLUMN IF EXISTS fx_rate_date,
    DROP COLUMN IF EXISTS value_basis,
    DROP COLUMN IF EXISTS reason_codes,
    DROP COLUMN IF EXISTS facts_state;

COMMIT;
