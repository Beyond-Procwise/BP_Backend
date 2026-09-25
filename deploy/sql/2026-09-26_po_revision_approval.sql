-- 2026-09-26 PO revisions and approval.
--
-- A purchase order is re-issued — "Revision 3", "Change Order 2" — and each issue is a
-- document in its own right: the Document match view (UI BACKEND_GAPS.md) must match an
-- invoice against the LATEST APPROVED revision, and list the earlier ones as Superseded.
-- Today nothing records either: po_id carries no revision (5,037 'PO999999' + 4
-- 'PO-9999-9999', no suffix of any kind), po_status is NULL on all 5,041 rows, and
-- purchase_order.yaml has no revision, change-order or approval field.
--
-- Revisions follow the convention quotes already use (context_layer.canonical_quote_revision):
-- a revised PO persists under its own key, "<po number> (Rev n)", revision 1 unsuffixed.
-- po_revision holds n as a number so nothing has to parse the id; approval_status holds
-- the approval the document states, normalised to approved | pending | rejected | cancelled.
-- NULL means "the document does not say", which is every PO extracted before this — the
-- gateway reads NULL as approved, exactly as before.
--
-- Additive and idempotent. Run BEFORE shipping the purchase_order.yaml fields:
-- load_all_schemas() checks every db_column against information_schema at start-up.
BEGIN;

ALTER TABLE proc.bp_purchase_order_raw  ADD COLUMN IF NOT EXISTS po_revision     INTEGER;
ALTER TABLE proc.bp_purchase_order_stg  ADD COLUMN IF NOT EXISTS po_revision     INTEGER;
ALTER TABLE proc.bp_purchase_order_trgt ADD COLUMN IF NOT EXISTS po_revision     INTEGER;

ALTER TABLE proc.bp_purchase_order_raw  ADD COLUMN IF NOT EXISTS approval_status TEXT;
ALTER TABLE proc.bp_purchase_order_stg  ADD COLUMN IF NOT EXISTS approval_status TEXT;
ALTER TABLE proc.bp_purchase_order_trgt ADD COLUMN IF NOT EXISTS approval_status TEXT;

-- The value set is closed so a typo cannot read as "not approved" and silently change which
-- revision is final.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'bp_purchase_order_trgt_approval_chk') THEN
    ALTER TABLE proc.bp_purchase_order_trgt ADD CONSTRAINT bp_purchase_order_trgt_approval_chk
      CHECK (approval_status IS NULL OR approval_status IN ('approved', 'pending', 'rejected', 'cancelled'));
  END IF;
END $$;

-- The Document match groups revisions by their base number on every read.
CREATE INDEX IF NOT EXISTS ix_bp_purchase_order_trgt_po_base
    ON proc.bp_purchase_order_trgt ((regexp_replace(po_id, '\s*\(\s*rev\M.*$', '', 'i')));

COMMIT;
