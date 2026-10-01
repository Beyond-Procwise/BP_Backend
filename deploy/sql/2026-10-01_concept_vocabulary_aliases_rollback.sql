-- Reverses 2026-10-01_concept_vocabulary_aliases.sql: removes exactly the two
-- aliases it added. Safe to run twice (array_remove of an absent value is a no-op).
BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'quotes')
 WHERE concept_code = 'doctype.quote';

UPDATE proc.bp_document_type
   SET aliases = array_remove(aliases, 'contracts')
 WHERE concept_code = 'doctype.contract_unspecified';

COMMIT;
