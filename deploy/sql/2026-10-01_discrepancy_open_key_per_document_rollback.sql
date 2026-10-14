-- Undo 2026-10-01_discrepancy_open_key_per_document.sql: back to the invoice-number key.
--
-- READ THIS BEFORE RUNNING IT. Reverting the index WITHOUT also reverting the two code sites
-- (persistence.write_discrepancies, promotion._DISCREPANCY_UPSERT) breaks every extraction
-- write: their ON CONFLICT clause names coalesce(source_file,''), and with no index of that
-- shape Postgres rejects the statement outright rather than degrading. Revert the code in the
-- same step, and restart the API.
--
-- The narrow index may also FAIL TO BUILD, and that is the fix working rather than a fault in
-- this script: once two documents sharing one number have each recorded their own finding,
-- those rows are legitimately distinct and the narrow key cannot hold both. If that happens,
-- the collision must be resolved by deciding which document's finding to keep -- which is
-- the data loss the forward migration exists to prevent. Do not resolve it by deleting rows
-- without a ruling.
--
-- Apply with: psql -v ON_ERROR_STOP=1 -f <this file>

BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_extraction_discrepancy_open_key;

CREATE UNIQUE INDEX ix_bp_extraction_discrepancy_open_key
    ON proc.bp_extraction_discrepancy
       (doc_type, coalesce(doc_pk_candidate, ''),
        issue_type, coalesce(field_name, ''))
 WHERE coalesce(status, 'open') <> 'resolved';

COMMIT;
