-- The buyer's signatory gets a column of its own.
--
-- A contract has TWO signatories, one per party. Until now proc.bp_contracts had
-- one pair of columns (contract_signatory_name / contract_signatory_role), so the
-- 2026-10-03 signature-block reader had to rule that the pair holds the
-- SUPPLIER's signatory -- "who bound the counterparty", the question a single slot
-- has to answer in a procurement system -- and the buyer's name was parsed and
-- then thrown away. On the one real contract in the corpus that discarded
-- "Sarah Johnson", signing for the Client, which the document states as plainly
-- as it states John Smith for the Marketer.
--
-- NAMING. The existing pair is NOT renamed to supplier_signatory_*. The name
-- `contract_signatory_name` is read by the gateway, the Obligations screen and
-- every consumer of proc.bp_contracts; renaming it is a breaking change for a
-- cosmetic gain, and a rename plus an addition in one migration cannot be rolled
-- back without deciding what to do with the data in between. So:
--
--     contract_signatory_name / contract_signatory_role   -> the SUPPLIER's
--     buyer_signatory_name    / buyer_signatory_role      -> the BUYER's
--
-- The asymmetry is documented in the column comments below rather than left for
-- a future reader to infer from one of the two names.
--
-- NOT NER-TYPED. extraction_schemas/contract.yaml declares these two fields with
-- ner_type_check: "none" on purpose. The entity sweep is exactly what stored
-- "Email Marketing" as the person who signed, and a PERSON-typed field with no
-- party-aware branch falls straight back into it. These columns are filled by
-- engineered/contract_signatories.py reading the signature block, or they stay
-- NULL.
--
-- Additive, idempotent, reversible. No index: nothing looks a contract up by who
-- signed it, and an unused index on a 0-row table is a liability, not a win.
BEGIN;

ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS buyer_signatory_name text,
    ADD COLUMN IF NOT EXISTS buyer_signatory_role text;

ALTER TABLE proc.bp_contracts
    ADD COLUMN IF NOT EXISTS buyer_signatory_name text,
    ADD COLUMN IF NOT EXISTS buyer_signatory_role text;

COMMENT ON COLUMN proc.bp_contracts.buyer_signatory_name IS
    'The person who signed for the BUYER, as the signature block names them. '
    'Read deterministically from the block (engineered/contract_signatories.py), '
    'never from the NER sweep, which once stored "Email Marketing" here-abouts. '
    'NULL when the document names no signatory for that party, which is the '
    'honest answer and not a gap.';
COMMENT ON COLUMN proc.bp_contracts.buyer_signatory_role IS
    'The job title beside the buyer signatory''s name ("Managing Director"), when '
    'the block carries one. Not the party role -- that is implied by the column.';
COMMENT ON COLUMN proc.bp_contracts.contract_signatory_name IS
    'The person who signed for the SUPPLIER. Named without a supplier_ prefix for '
    'historical reasons: it predates buyer_signatory_name and is read by the '
    'gateway and the Obligations screen, so renaming it would be a breaking '
    'change. See 2026-10-04_contract_buyer_signatory.sql.';

COMMIT;
