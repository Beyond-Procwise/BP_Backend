-- Reverses 2026-10-01_concept_vocabulary_restored_aliases.sql: removes exactly
-- the two aliases it added. Safe to run twice (array_remove of an absent value
-- is a no-op).
--
-- Exact only for a database that actually took the forward migration. It removes
-- 'framework'/'notice' unconditionally, so:
--   * if either alias was added by hand before the forward run (which then
--     no-opped), this rollback still strips it;
--   * on a database built fresh from the new seed.py the aliases are seeded
--     already, so this rollback removes seeded aliases.
-- It does NOT touch 'framework agreement', 'framework contract' or
-- 'general notice'. Accepted for a two-alias change; know it before running.
--
-- Afterwards an upload declaring the bare category 'framework' or 'notice' is
-- refused as an unrecognised category again, which is what Task 1 shipped.
BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'framework')
 WHERE concept_code = 'doctype.framework_agreement';

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'notice')
 WHERE concept_code = 'doctype.notice_general';

COMMIT;
