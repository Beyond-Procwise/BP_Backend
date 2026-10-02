-- Two columns on proc.bp_document_type, so the stand-down rule is DATA.
--
-- A structure whose purpose is to sit beneath a parent should only claim a page
-- that actually names that parent. 'order form' is the measured case: it titles
-- every quote-template workbook on this corpus, and added as a plain structure
-- it flips 13 quote documents to 'disagreed' with no true positive
-- (specs/2026-10-02-contract-structures-design.md §4).
--
-- requires_parent_evidence  -- does this structure have to show its parent?
-- parent_evidence_phrases   -- the matchable phrases that count as showing it.
--
-- Phrases, not prose. The existing structural_signals are sentences
-- ("lists incorporated documents") and cannot match a page -- open item 5 of
-- specs/2026-10-01-document-relationship-layer-rulings.md. These are compared
-- with the same fold() normalisation every alias uses.
--
-- NO ROW SETS THE FLAG HERE. Behaviour is unchanged by this migration; the
-- order-form row arrives in 2026-10-02_document_type_order_form_sales_order.sql.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_document_type
    ADD COLUMN IF NOT EXISTS requires_parent_evidence boolean NOT NULL DEFAULT false;

ALTER TABLE proc.bp_document_type
    ADD COLUMN IF NOT EXISTS parent_evidence_phrases text[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN proc.bp_document_type.requires_parent_evidence IS
    'When true, this structure only claims a document that names its parent or '
    'states an order of precedence. Intended for doctype.order_form, which the '
    'order-form migration sets; no row sets it as of this migration. The other '
    'child structures (call-off, SOW, schedule) are conceptually the same but '
    'their current behaviour is measured and no evidence calls for changing it.';
COMMENT ON COLUMN proc.bp_document_type.parent_evidence_phrases IS
    'Matchable phrases that count as naming a parent. Compared with the same '
    'fold() normalisation as aliases. Prose belongs in structural_signals.';

COMMIT;
