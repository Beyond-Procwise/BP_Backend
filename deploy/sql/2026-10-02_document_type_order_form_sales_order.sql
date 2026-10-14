-- The two structures Nick named on 2026-10-02: order form and sales order.
--
-- ORDER FORM comes back as a STRUCTURE IN ITS OWN RIGHT, not as an alias of
-- doctype.call_off_contract. It was dropped as that alias on 2026-10-01 because
-- it titles every quote-template workbook on this corpus: 12 false
-- disagreements out of 12 uses, no true positive
-- (2026-10-02_concept_vocabulary_drop_order_form_alias.sql).
--
-- What makes it safe now is requires_parent_evidence, added by
-- 2026-10-02_document_type_parent_evidence.sql. An order form only claims a
-- document that names the agreement it sits under or states an order of
-- precedence. Measured over the 53 corpus documents with stored parsed text: without
-- the rule, 13 quote workbooks flip to 'disagreed'; with it, every document
-- resolves exactly as it does today. This is the build spec's own distinction --
-- "'order form' means one thing under a framework and another on its own".
--
-- THIS MIGRATION IS WHERE THAT RULE BECOMES LIVE BEHAVIOUR. doctype.order_form
-- is the first and only row to carry the flag. Do not apply it before
-- 2026-10-02_document_type_parent_evidence.sql: without the column the structure
-- claims every page carrying the words.
--
-- SALES ORDER routes at the PURCHASE_ORDER pipeline, not the contract pipeline
-- (Nick's ruling, 2026-10-02): it is the supplier's mirror of a purchase order
-- and carries lines, quantities and a total, so the PO schema is the one that
-- fits it. pipeline_doc_type only takes effect when an uploader types that
-- category; a document dropped in the Contracts zone still routes on 'contract'.
--
-- Consequence to know before running: an alias is also an acceptable UPLOAD
-- CATEGORY (src/services/concepts/routing.py). After this,
--   'order form'  -> ('contract', 'doctype.order_form')
--   'sales order' -> ('purchase_order', 'doctype.sales_order')
-- Neither raised before; both now route.
--
-- APPEND ORDER IS PART OF THE DATA: the full-column drift test compares
-- not_to_be_confused_with and aliases as ORDERED lists against seed.py. The two
-- new pointers appended to existing concepts go LAST in both places.
--
-- Additive (ON CONFLICT DO NOTHING), idempotent, reversible.
-- Order matters: bp_document_type's foreign keys need the concepts first.
BEGIN;

INSERT INTO proc.bp_concept
    (concept_code, domain, definition, not_to_be_confused_with, status, source, rejection_reason)
VALUES
    ('doctype.order_form', 'DOCUMENT_TYPE',
     'Orders specific goods or services on the terms of an agreement it names.',
     ARRAY['doctype.call_off_contract','doctype.quote','doctype.order']::text[],
     'active', 'seed', NULL),
    ('doctype.sales_order', 'DOCUMENT_TYPE',
     'The supplier''s own confirmation of an order it has received.',
     ARRAY['doctype.order','doctype.invoice']::text[],
     'active', 'seed', NULL)
ON CONFLICT (concept_code) DO NOTHING;

-- The table exists to record what a term is mistaken for, so the two
-- already-seeded concepts most at risk gain a pointer at the new ones.
UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_append(not_to_be_confused_with, 'doctype.order_form')
 WHERE concept_code = 'doctype.call_off_contract'
   AND NOT (not_to_be_confused_with @> ARRAY['doctype.order_form']::text[]);

UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_append(not_to_be_confused_with, 'doctype.sales_order')
 WHERE concept_code = 'doctype.order'
   AND NOT (not_to_be_confused_with @> ARRAY['doctype.sales_order']::text[]);

INSERT INTO proc.bp_document_type
    (concept_code, role, default_parent_type, execution_mode, aliases, identifiers,
     structural_signals, pipeline_doc_type, status, source,
     requires_parent_evidence, parent_evidence_phrases)
VALUES
    ('doctype.order_form', 'role.master', 'doctype.framework_agreement', 'exec.bilateral',
     ARRAY['order form']::text[],
     '[{"field": "framework_ref", "pattern": null, "parent_type": "doctype.framework_agreement"}]'::jsonb,
     ARRAY['lists incorporated documents','states an order of precedence']::text[],
     'contract', 'active', 'seed',
     true,
     -- Phrases, not prose: these are compared against the page with the same
     -- fold() normalisation as an alias, so 'Call-Off' satisfies 'call off'.
     -- 'incorporated' alone is NOT here, deliberately. Phrases match whole-word
     -- (type_resolver._names_a_parent uses _find_all), and bare 'incorporated'
     -- still matches a SUPPLIER NAME -- "Acme Incorporated" -- which is a hole
     -- straight back into the 13-workbook defect this rule exists to close.
     -- Measured 2026-10-02: substring matching claimed 3 false pages; whole-word
     -- matching fixed 2 of them; narrowing this phrase fixed the third, and all
     -- five genuine order-form shapes still match.
     ARRAY['framework','order of precedence',
           'incorporated into','incorporated by reference','call off',
           'framework agreement no','framework agreement number','framework agreement ref',
           'master agreement no','master agreement number','master agreement ref',
           'parent agreement no','parent contract no','principal agreement no']::text[]),
    ('doctype.sales_order', 'role.transaction', 'doctype.order', 'exec.unilateral',
     ARRAY['sales order','sales order acknowledgement','order acknowledgement']::text[],
     '[{"field": "po_id", "pattern": null, "parent_type": "doctype.order"}]'::jsonb,
     ARRAY['line items with quantities and a total']::text[],
     'purchase_order', 'active', 'seed',
     false, ARRAY[]::text[])
ON CONFLICT (concept_code) DO NOTHING;

COMMIT;
