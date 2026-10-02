-- Removes exactly the two columns 2026-10-02_document_type_parent_evidence.sql adds.
-- Run the order_form/sales_order rollback FIRST if that migration has been applied:
-- dropping these columns while doctype.order_form relies on the flag would make
-- 'order form' claim every quote-template workbook again.
BEGIN;

ALTER TABLE proc.bp_document_type DROP COLUMN IF EXISTS parent_evidence_phrases;
ALTER TABLE proc.bp_document_type DROP COLUMN IF EXISTS requires_parent_evidence;

COMMIT;
