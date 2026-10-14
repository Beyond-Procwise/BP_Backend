-- Rollback of 2026-10-04_bp_demand_id_sequence.sql.
--
-- Dropping the sequence loses no demand: the ids already minted from it are plain text in
-- bp_demand.demand_id and stay exactly as they are. What it does mean is that the gateway has
-- nothing to mint from, so re-run this only together with reverting the code that uses it.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_demand_status;
DROP SEQUENCE IF EXISTS proc.bp_demand_id_seq;
COMMIT;
