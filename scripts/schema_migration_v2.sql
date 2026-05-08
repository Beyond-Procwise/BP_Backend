-- Extraction Pipeline v2 Schema Migration
-- Adds missing financial columns to PO, fixes typo, creates vendor_profile table

-- PO header: add tax/total columns to match Invoice/Quote pattern
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS tax_percent numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS tax_amount numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS total_amount_incl_tax numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS supplier_id text;

-- Fix typo in PO line items (unit_of_measue → unit_of_measure)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='proc' AND table_name='bp_po_line_items'
               AND column_name='unit_of_measue') THEN
        ALTER TABLE proc.bp_po_line_items RENAME COLUMN unit_of_measue TO unit_of_measure;
    END IF;
END $$;

-- Vendor profile table for extraction pattern learning
CREATE TABLE IF NOT EXISTS proc.vendor_profile (
    supplier_name text PRIMARY KEY,
    profile_data jsonb NOT NULL DEFAULT '{}',
    extraction_count integer DEFAULT 0,
    avg_confidence numeric DEFAULT 0,
    last_extraction timestamp,
    created_date timestamp DEFAULT NOW(),
    last_modified_date timestamp DEFAULT NOW()
);
