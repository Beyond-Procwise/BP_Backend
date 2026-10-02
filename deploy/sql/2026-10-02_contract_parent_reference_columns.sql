-- Two reference columns the vocabulary has always declared and the schema never
-- had a field for.
--
-- proc.bp_document_type says doctype.call_off_contract points at its framework
-- through a field called framework_ref, and doctype.order_form does the same.
-- No such field existed in extraction_schemas/contract.yaml, so the pointer was
-- declared and never filled.
--
-- SEPARATE FROM parent_contract_id, which stays as it is. A SOW names the master
-- agreement it sits under without amending anything; parent_contract_id anchors
-- on 'amends', 'supplements', 'varies'. "The document this one sits under" and
-- "the document this one changes" are different facts, and the hierarchy maths
-- needs to tell a child from an amendment.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS framework_ref        text,
    ADD COLUMN IF NOT EXISTS parent_agreement_ref text;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS framework_ref        text,
    ADD COLUMN IF NOT EXISTS parent_agreement_ref text;

COMMENT ON COLUMN proc.bp_contracts.framework_ref IS
    'The framework agreement this document is called off under, as the document '
    'states it. A reference as printed, never a resolved key -- 1,561 existing '
    'parent_contract_id values resolve to 0 real contracts, which is why nothing '
    'auto-links on a reference alone.';
COMMENT ON COLUMN proc.bp_contracts.parent_agreement_ref IS
    'The agreement this document sits under without amending, as the document '
    'states it. parent_contract_id carries the amends/supplements/varies case.';

COMMIT;
