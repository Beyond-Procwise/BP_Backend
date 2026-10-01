-- Undo 2026-10-01_sqldb_discrepancy_open_key.sql.
--
-- This is reversible only because the forward migration RESOLVED the duplicates instead of
-- deleting them. Every row it set aside is still present and still carries its own figures;
-- resolved_by = 'sqldb-open-key-migration' is the tag that identifies exactly those rows and
-- nothing else.
--
-- ORDER MATTERS. The index must go FIRST: reopening the duplicates puts them back inside the
-- index predicate, and while a unique index is still in place the UPDATE would fail on the
-- first key that regains a second open row.
--
-- Apply with: psql -v ON_ERROR_STOP=1 -f <this file>

BEGIN;

DROP INDEX IF EXISTS proc.ix_bp_extraction_discrepancy_open_key;

UPDATE proc.bp_extraction_discrepancy
   SET status            = 'open',
       resolved_at       = NULL,
       resolution_action = NULL,
       resolved_by       = NULL,
       notes             = nullif(
           regexp_replace(
               coalesce(notes, ''),
               ' \[superseded: a later finding holds this key; set aside, not deleted, so the differing figures stay inspectable\]',
               ''),
           '')
 WHERE resolved_by = 'sqldb-open-key-migration';

COMMIT;
