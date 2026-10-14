-- Reverses 2026-10-01_concept_vocabulary_aliases.sql: removes exactly the two
-- aliases it added. Safe to run twice (array_remove of an absent value is a no-op).
--
-- Exact only for a database that actually took the forward migration. It
-- removes 'quotes'/'contracts' unconditionally, so:
--   * if either alias was added by hand before the forward run (which then
--     no-opped), this rollback still strips it;
--   * on a database built fresh from the new seed.py the aliases are seeded
--     already, so this rollback removes seeded aliases.
-- Accepted for a two-alias change; know it before running.
BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'quotes')
 WHERE concept_code = 'doctype.quote';

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'contracts')
 WHERE concept_code = 'doctype.contract_unspecified';

COMMIT;
