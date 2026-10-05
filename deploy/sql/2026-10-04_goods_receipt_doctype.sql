-- The vocabulary learns the goods receipt.
--
-- Order matters twice over:
--   1. bp_document_type's foreign keys need the concept first, so bp_concept
--      is inserted before bp_document_type (as
--      2026-10-02_document_type_order_form_sales_order.sql does);
--   2. deploy/sql/2026-10-04_goods_receipt_tables.sql MUST already be applied.
--      pipeline_doc_type='goods_receipt' routes a document at
--      proc.bp_goods_receipt_{raw,stg,trgt}; without them the failure surfaces
--      as a SQL error mid-extraction. src/services/concepts/validate.py's
--      _PIPELINES is the matching code-side gate.
--
-- ALIAS ORDER IS PART OF THE DATA. src/services/concepts/seed.py lists these
-- eleven in exactly this order and
-- tests/services/concepts/test_concept_table.py::test_document_type_rows_equal_the_seed_column_for_column
-- compares them as an ORDERED list. New aliases go last, in both places.
--
-- Consequence to know before running: an alias is also an acceptable UPLOAD
-- CATEGORY (src/services/concepts/routing.py). After this, 'delivery note',
-- 'grn', 'packing slip' and the rest route at the goods_receipt pipeline when
-- an uploader types them. None of the eleven was owned by any other type when
-- this was written -- tests/extraction/test_goods_receipt_classification_baseline.py
-- ::test_every_proposed_alias_is_unowned is the standing proof of that.
--
-- Additive (ON CONFLICT DO NOTHING), idempotent, reversible.
BEGIN;

-- The DATABASE has its own copy of the pipeline whitelist, added by
-- 2026-10-01_concept_vocabulary.sql, and it rejected this row before the insert
-- was ever reached. Widen it here, in the same transaction as the row it is
-- widened for, so the two can never be out of step: code-side _PIPELINES, this
-- constraint and the six physical tables are one fact in three places.
ALTER TABLE proc.bp_document_type DROP CONSTRAINT IF EXISTS ck_bp_document_type_pipeline;
ALTER TABLE proc.bp_document_type ADD CONSTRAINT ck_bp_document_type_pipeline CHECK (
    pipeline_doc_type IS NULL
    OR pipeline_doc_type = ANY (ARRAY['invoice','purchase_order','quote','contract','goods_receipt'])
);

INSERT INTO proc.bp_concept
    (concept_code, domain, definition, not_to_be_confused_with, status, source, rejection_reason)
VALUES
    ('doctype.goods_receipt', 'DOCUMENT_TYPE',
     'Records what was physically delivered and accepted.',
     ARRAY[]::text[], 'active', 'seed', NULL)
ON CONFLICT (concept_code) DO NOTHING;

INSERT INTO proc.bp_document_type
    (concept_code, role, default_parent_type, execution_mode, aliases, identifiers,
     structural_signals, pipeline_doc_type, status, source,
     requires_parent_evidence, parent_evidence_phrases)
VALUES
    ('doctype.goods_receipt', 'role.transaction', 'doctype.order', 'exec.unilateral',
     ARRAY['goods receipt','goods received note','grn','delivery note',
           'despatch note','dispatch note','advice note','packing list',
           'packing slip','proof of delivery','pod']::text[],
     '[{"field": "grn_id", "pattern": null, "parent_type": null},
       {"field": "po_id", "pattern": null, "parent_type": "doctype.order"}]'::jsonb,
     ARRAY['quantities with no prices','signed for on receipt',
           'a carrier, vehicle or consignment reference']::text[],
     'goods_receipt', 'active', 'seed',
     false, ARRAY[]::text[])
ON CONFLICT (concept_code) DO NOTHING;

COMMIT;
