-- The open-findings key identifies a finding by DOCUMENT, not by invoice number.
--
-- THE BUG. The key was (doc_type, doc_pk_candidate, issue_type, field_name).
-- `doc_pk_candidate` is the invoice/PO/quote number read OUT of the document; it is not
-- an identifier OF the document. So two DIFFERENT documents carrying the same number
-- collided, and the later one's ON CONFLICT DO UPDATE overwrote the earlier one's finding
-- in place. Nothing recorded that the overwritten finding had ever existed.
--
-- Proved on bp_testdb before writing this, in a rolled-back transaction: write a
-- sum_mismatch for PO1_IT_Invoice_LowerCost.pdf (15,877.50 against an expected 15,757.50),
-- then one for MASTER Invoice for PO1.pdf (17,065.50 against 16,945.50) under the same
-- invoice number. One row survives, holding only MASTER's figures.
--
-- It has already happened in live data. bp_testdb holds two documents under invoice
-- 01-2024-002 -- PO1_IT_Invoice_LowerCost.pdf (promoted 2026-07-31) and MASTER Invoice for
-- PO1.pdf (promoted 2026-10-01) -- and findings for only the second. And the canonical real
-- case for one invoice number arriving on two documents is the same invoice submitted
-- twice, which is what duplicate_invoice_detector.py exists to catch: the findings layer
-- was keeping only the last of them.
--
-- Half of this was already understood in the code. persistence._type_finding_pk synthesises
-- a fake 'file:<path>' key for documents with NO pk, with the comment "so two pk-less
-- documents do not share the open-row identity and overwrite each other". Documents WITH a
-- real number were left colliding. This generalises that fix instead of special-casing it.
--
-- WHY source_file IS THE RIGHT DISCRIMINATOR, and the property this must not cost. The key
-- exists so that RE-READING one document (new raw_id, same findings) refreshes rather than
-- stacks -- the Test Data_300726 incident, 94 duplicated keys. A re-read keeps the same
-- source_file, so refresh still happens; a different document brings a different
-- source_file, so it gets its own row. raw_id could NOT be used: it changes on every
-- re-read, which would restore the stacking bug exactly.
--
-- Checked on both databases before writing this:
--   * no document is recorded under two different source_file spellings (0 on each), so
--     widening the key cannot split one document's findings in two;
--   * source_file is never null or blank on bp_testdb; 8 unresolved rows on bp_sqldb have
--     it blank, which coalesce(source_file,'') folds together exactly as the existing
--     coalesce on doc_pk_candidate and field_name already does;
--   * zero duplicate keys under the NEW key on either database, so the index builds without
--     touching a single row. A wider key can only split groups, never merge them -- which
--     is why this migration has no data step at all.
--
-- RESIDUAL RISK, stated: if a document's path ever changes between runs -- a re-upload under
-- a different prefix -- its findings would stack again. Nothing does that today. Related and
-- not fixed here: 834 of 5,373 source_file values on bp_testdb are paths and the rest bare
-- filenames, so two writers disagree on spelling. No single document is spelled both ways,
-- so it does not bite; if that ever changes, normalise on write rather than widening again.
--
-- DEPLOYMENT ORDER MATTERS. The clause and the index are matched by SHAPE, and Postgres does
-- not warn on a mismatch -- it rejects every write with "there is no unique or exclusion
-- constraint matching the ON CONFLICT specification". That is how bp_sqldb silently recorded
-- nothing for two months. So this migration and the two code sites
-- (persistence.write_discrepancies, promotion._DISCREPANCY_UPSERT) must land together, and
-- the API restarted. Between the index swap and the restart, writes from the old code fail.
--
-- Apply with: psql -v ON_ERROR_STOP=1 -f <this file>
-- Apply to BOTH bp_testdb and bp_sqldb.

BEGIN;

-- The old index must GO, not merely be supplemented: while it exists it keeps enforcing the
-- too-narrow uniqueness, which is the very thing that refuses a second document's finding.
DROP INDEX IF EXISTS proc.ix_bp_extraction_discrepancy_open_key;

CREATE UNIQUE INDEX ix_bp_extraction_discrepancy_open_key
    ON proc.bp_extraction_discrepancy
       (doc_type, coalesce(doc_pk_candidate, ''), coalesce(source_file, ''),
        issue_type, coalesce(field_name, ''))
 WHERE coalesce(status, 'open') <> 'resolved';

-- Refuse to finish unless the index is really there under the name both code sites' clauses
-- will be matched against. A migration that reports success while every write is rejected is
-- the failure this project has already paid for once.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = 'proc'
           AND indexname = 'ix_bp_extraction_discrepancy_open_key'
           AND indexdef LIKE '%source_file%') THEN
        RAISE EXCEPTION
            'the per-document open key is not in place; extraction writes would be rejected';
    END IF;
END $$;

COMMIT;
