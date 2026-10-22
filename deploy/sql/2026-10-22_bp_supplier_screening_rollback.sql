-- Rollback for 2026-10-22_bp_supplier_screening.sql. Drops the register and everything recorded in it:
-- run only if the table was created by that file and nobody has recorded a check yet (SELECT count(*) first).
BEGIN;
DROP TRIGGER IF EXISTS tr_bp_supplier_screening_append_only ON proc.bp_supplier_screening;
DROP FUNCTION IF EXISTS proc.bp_supplier_screening_append_only();
DROP TABLE IF EXISTS proc.bp_supplier_screening;
COMMIT;
