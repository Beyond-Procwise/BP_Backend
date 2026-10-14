-- Drop "order form" as an alias of doctype.call_off_contract.
--
-- WHY: "order form" is a genuine name for a call-off contract, which is why it was
-- seeded. On this corpus it is also the title cell that every quote-template
-- workbook carries, so it produced 12 disagreements out of 12 uses and no true
-- positive: twelve quote-category documents (Aureus, Lattice, Meridia, Orbis) each
-- reported evidence of doctype.call_off_contract against their declared quote.
-- Measured before this migration: 38 agreed / 12 disagreed over 50 live documents;
-- after: 50 agreed / 0 disagreed. A real call-off still matches on
-- "call-off contract", "call off contract" and "call-off".
--
-- Re-adding it needs a way to distinguish a quote template's "Order Form" heading
-- from a real call-off's, which needs the golden-set documents that do not exist yet.
-- See specs/2026-10-01-document-relationship-layer-rulings.md.
--
-- NOTE: an alias is also an acceptable UPLOAD CATEGORY, because the upload gate and
-- the classifier read the same column. So this also stops "order form" being accepted
-- as a category by proc.process_monitor ingestion. No live document declares it --
-- the only categories in use are quote, invoice and po -- so nothing that routes
-- today stops routing.
--
-- Additive in the sense this project means it: idempotent, reversible, and scoped to
-- one concept_code. Safe to run twice. Apply to bp_testdb AND bp_sqldb.

BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'order form')
 WHERE concept_code = 'doctype.call_off_contract'
   AND aliases @> ARRAY['order form']::text[];

COMMIT;
