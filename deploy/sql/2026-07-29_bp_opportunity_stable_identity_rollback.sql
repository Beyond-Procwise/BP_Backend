-- Rollback for 2026-07-29_bp_opportunity_stable_identity.sql
--
-- Drops the stable-identity index and the retired_at column. Rows that the
-- deduplication step removed are NOT restored — that step collapsed duplicates
-- the old keying had already created, and the surviving row is the current one.
BEGIN;

DROP INDEX IF EXISTS proc.ux_bp_opportunity_ref;
DROP INDEX IF EXISTS proc.ix_bp_opportunity_retired_at;

ALTER TABLE proc.bp_opportunity
    ALTER COLUMN opportunity_ref_id DROP NOT NULL;

ALTER TABLE proc.bp_opportunity
    DROP COLUMN IF EXISTS retired_at;

COMMIT;
