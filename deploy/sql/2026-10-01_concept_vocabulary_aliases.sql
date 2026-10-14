-- Adds the plural spellings the legacy upload path accepted
-- (utils.procurement_schema.CATEGORY_TO_DOC_TYPE has 'quotes' and 'contracts')
-- to the document-type vocabulary. The original seed used ON CONFLICT DO NOTHING,
-- so editing it would not reach a database that already holds the rows.
--
-- Additive, idempotent (an alias already present is not appended), reversible.
BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_append(aliases, 'quotes')
 WHERE concept_code = 'doctype.quote'
   AND NOT (aliases @> ARRAY['quotes']::text[]);

UPDATE proc.bp_document_type
   SET aliases = array_append(aliases, 'contracts')
 WHERE concept_code = 'doctype.contract_unspecified'
   AND NOT (aliases @> ARRAY['contracts']::text[]);

COMMIT;
