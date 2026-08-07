-- Rollback for 2026-08-07_uom_canonical.sql
--
-- ORDERING WARNING. Drop the indexes before the table. DROP TABLE removes its
-- own indexes implicitly, so doing it the other way round succeeds once and
-- then fails on a re-run when DROP INDEX finds nothing but the table is gone --
-- which is how a rollback stops being idempotent exactly when it is most
-- likely to be re-run.
--
-- DESTRUCTIVE. This discards any human confirmations recorded against
-- proposed units (confirmed_by / confirmed_at) and the observation counts. The
-- seeded rows are reproducible from the forward migration; the confirmations
-- are not. Export them first if they matter:
--
--   \copy (SELECT * FROM proc.bp_uom_canonical WHERE confirmed_by IS NOT NULL)
--     TO 'uom_confirmations.csv' CSV HEADER
BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_uom_canonical_dimension;
DROP INDEX IF EXISTS proc.ix_bp_uom_canonical_status;

DROP TABLE IF EXISTS proc.bp_uom_canonical;

COMMIT;
