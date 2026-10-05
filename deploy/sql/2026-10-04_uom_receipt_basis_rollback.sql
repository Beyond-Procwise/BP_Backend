-- Reverses deploy/sql/2026-10-04_uom_receipt_basis.sql.
BEGIN;
ALTER TABLE proc.bp_uom_canonical DROP CONSTRAINT IF EXISTS ck_bp_uom_canonical_receipt_basis;
ALTER TABLE proc.bp_uom_canonical ALTER COLUMN receipt_basis DROP DEFAULT;
ALTER TABLE proc.bp_uom_canonical DROP COLUMN IF EXISTS receipt_basis;
COMMIT;
