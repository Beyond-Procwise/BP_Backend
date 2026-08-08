-- Phase 1b: structured columns on proc.bp_opportunity.
--
-- calculation_details becomes derived and optional. These columns are the
-- system of record for the numbers a finding rests on; the JSONB stays written
-- for one release behind a logged deprecation shim.
--
-- Deliberately NOT back-filled by this migration. What can honestly be
-- harvested from the existing JSONB is narrow, and pretending otherwise in SQL
-- would bury the judgement. scripts/backfill_opportunity_structured.py does it
-- and records what it could not resolve.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_opportunity
    ADD COLUMN IF NOT EXISTS currency       text,
    ADD COLUMN IF NOT EXISTS amount_native  numeric,
    ADD COLUMN IF NOT EXISTS unit_price     numeric,
    ADD COLUMN IF NOT EXISTS quantity       numeric,
    ADD COLUMN IF NOT EXISTS uom            text,
    ADD COLUMN IF NOT EXISTS uom_normalised text,
    ADD COLUMN IF NOT EXISTS fx_rate        numeric,
    ADD COLUMN IF NOT EXISTS fx_rate_date   timestamptz,
    ADD COLUMN IF NOT EXISTS value_basis    text,
    ADD COLUMN IF NOT EXISTS reason_codes   text[],
    -- RESOLVED: at least one number was read out of the JSONB into a column.
    -- INDETERMINATE: nothing structured was present. NOT a synonym for "zero"
    -- and never inferred -- an unparseable finding keeps every column NULL and
    -- says so, rather than presenting a guess as a harvested value.
    ADD COLUMN IF NOT EXISTS facts_state    text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_bp_opportunity_facts_state'
          AND conrelid = 'proc.bp_opportunity'::regclass
    ) THEN
        ALTER TABLE proc.bp_opportunity
            ADD CONSTRAINT ck_bp_opportunity_facts_state
            CHECK (facts_state IS NULL OR facts_state IN ('RESOLVED', 'INDETERMINATE'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_facts_state
    ON proc.bp_opportunity (facts_state);

COMMIT;
