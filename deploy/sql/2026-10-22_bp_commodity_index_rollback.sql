-- Rollback for 2026-10-22_bp_commodity_index.sql. Drops the register and every series loaded into it:
-- run only if the table was created by that file and nothing has been loaded yet (SELECT count(*) first).
BEGIN;
DROP TRIGGER IF EXISTS tr_bp_commodity_index_append_only ON proc.bp_commodity_index;
DROP FUNCTION IF EXISTS proc.bp_commodity_index_append_only();
DROP TABLE IF EXISTS proc.bp_commodity_index;
COMMIT;
