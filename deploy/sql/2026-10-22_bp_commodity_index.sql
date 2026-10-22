-- The commodity index register: published index series the organisation loads (LME, PPI, energy,
-- labour ...), one row per series code and period, as published, with who loaded it and from where.
-- FOR REVIEW. NOT APPLIED to any database. Idempotent; one transaction. Rollback: the _rollback.sql beside it.
--
-- Why a register and not a feed: the platform subscribes to no index publisher, and four skills
-- (negotiation planning, pricing mechanisms, savings methodology, renewal) need a published series with a
-- code, publisher and date before they may split a price increase, neutralise a saving or design an
-- index-linked formula. A person loads the series here (POST /commodity-index/{series_code}); the agent
-- tool commodity_index_lookup reads it and computes only the move between two recorded points. No series
-- means "index history not available", never an estimate. A publisher feed, when one exists, writes into
-- this same table.
--
-- Append-only: a corrected value is loaded again and supersedes the earlier row for that period by
-- created_date; nothing is edited or deleted, so what a formula rested on can always be shown.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_commodity_index (
    index_id         BIGSERIAL PRIMARY KEY,
    series_code      TEXT NOT NULL,          -- the publisher's code, e.g. 'LME-AL-CASH', 'ONS-PPI-6.1'
    publisher        TEXT NOT NULL,          -- who publishes it, e.g. 'LME', 'ONS', 'Eurostat'
    description      TEXT,                   -- what it measures, in the publisher's words
    currency         TEXT,                   -- for a price series; NULL for a pure index
    unit             TEXT,                   -- e.g. 'USD/tonne', 'index 2015=100'
    period_date      DATE NOT NULL,          -- the period the value is for (first day of the month for monthly)
    value            NUMERIC NOT NULL,
    source_reference TEXT,                   -- where it was taken from: a URL, a report, a file name
    recorded_by      TEXT NOT NULL,
    created_date     TIMESTAMPTZ NOT NULL DEFAULT now(),
    tenant_id        TEXT NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS ix_bp_commodity_index_series_period
    ON proc.bp_commodity_index (series_code, period_date DESC, created_date DESC);

CREATE OR REPLACE FUNCTION proc.bp_commodity_index_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_commodity_index is append-only: load the corrected value again instead of changing %',
        COALESCE(OLD.index_id::text, '?');
END $$;

DROP TRIGGER IF EXISTS tr_bp_commodity_index_append_only ON proc.bp_commodity_index;
CREATE TRIGGER tr_bp_commodity_index_append_only
    BEFORE UPDATE OR DELETE ON proc.bp_commodity_index
    FOR EACH ROW EXECUTE FUNCTION proc.bp_commodity_index_append_only();

COMMIT;
