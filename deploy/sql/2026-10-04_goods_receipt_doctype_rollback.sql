-- Reverses deploy/sql/2026-10-04_goods_receipt_doctype.sql.
-- Order matters: the document type carries the foreign key onto the concept,
-- and the check constraint cannot be narrowed while a row still uses the fifth
-- value, so the row goes before the constraint.
BEGIN;
DELETE FROM proc.bp_document_type WHERE concept_code = 'doctype.goods_receipt';
DELETE FROM proc.bp_concept       WHERE concept_code = 'doctype.goods_receipt';

ALTER TABLE proc.bp_document_type DROP CONSTRAINT IF EXISTS ck_bp_document_type_pipeline;
ALTER TABLE proc.bp_document_type ADD CONSTRAINT ck_bp_document_type_pipeline CHECK (
    pipeline_doc_type IS NULL
    OR pipeline_doc_type = ANY (ARRAY['invoice','purchase_order','quote','contract'])
);
COMMIT;
