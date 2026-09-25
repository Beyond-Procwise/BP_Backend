-- Rollback for 2026-09-26_po_revision_approval.sql.
-- Remove the po_revision / approval_status fields from purchase_order.yaml FIRST:
-- load_all_schemas() refuses to start when a declared db_column is missing.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_purchase_order_trgt_po_base;
ALTER TABLE proc.bp_purchase_order_trgt DROP CONSTRAINT IF EXISTS bp_purchase_order_trgt_approval_chk;
ALTER TABLE proc.bp_purchase_order_trgt DROP COLUMN IF EXISTS approval_status;
ALTER TABLE proc.bp_purchase_order_stg  DROP COLUMN IF EXISTS approval_status;
ALTER TABLE proc.bp_purchase_order_raw  DROP COLUMN IF EXISTS approval_status;
ALTER TABLE proc.bp_purchase_order_trgt DROP COLUMN IF EXISTS po_revision;
ALTER TABLE proc.bp_purchase_order_stg  DROP COLUMN IF EXISTS po_revision;
ALTER TABLE proc.bp_purchase_order_raw  DROP COLUMN IF EXISTS po_revision;
COMMIT;
