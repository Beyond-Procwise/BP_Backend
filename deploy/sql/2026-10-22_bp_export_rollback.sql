-- Rollback for 2026-10-22_bp_export.sql. Drops the exports table and every file in it:
-- run only if the table was created by that file (SELECT count(*) first; exports are rebuildable).
BEGIN;
DROP TABLE IF EXISTS proc.bp_export;
COMMIT;
