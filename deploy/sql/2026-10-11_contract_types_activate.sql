-- Switches on the four contract types seeded 'proposed' on 2026-10-08 (DPA, side
-- letter, renewal agreement, guaranty) and reads them through the contract pipeline.
--
-- Why 'contract': each is a KIND of contract. With pipeline_doc_type NULL a true
-- DPA uploaded to the contract zone was "recognised, but nothing ingests it", so
-- it could not be a refinement of the generic 'contract' declaration.
--
-- Safe only because all four are title_only (2026-10-09_document_type_title_only.sql):
-- their names are everyday clause wording, and without that flag two body mentions
-- relabelled an untitled contract. The DO block refuses to run if the flag is
-- missing on any of them, so this can never be applied ahead of that migration.
--
-- BOTH tables: `status` lives on proc.bp_concept AND proc.bp_document_type for the
-- same type, and a type whose concept is still 'proposed' is dropped by the loader.
-- execution_mode is left NULL: nothing reads it and none was ruled on.
--
-- Deploy order: with or after the code (seed.py); either order is harmless, since
-- the running service reads these tables, but the seed-vs-table test compares them.
-- The loader notices the change by max(recorded_at), which is why it is bumped.
--
-- Idempotent, reversible (2026-10-11_contract_types_activate_rollback.sql).
BEGIN;

DO $$
BEGIN
    IF (SELECT count(*) FROM proc.bp_document_type
         WHERE concept_code IN ('doctype.dpa', 'doctype.side_letter',
                                'doctype.renewal', 'doctype.guaranty')
           AND title_only IS TRUE) <> 4 THEN
        RAISE EXCEPTION 'the four contract types are not all title_only; apply '
                        '2026-10-09_document_type_title_only.sql first';
    END IF;
END $$;

UPDATE proc.bp_concept
   SET status = 'active', recorded_at = now()
 WHERE concept_code IN ('doctype.dpa', 'doctype.side_letter',
                        'doctype.renewal', 'doctype.guaranty')
   AND status = 'proposed';

UPDATE proc.bp_document_type
   SET status = 'active', pipeline_doc_type = 'contract', recorded_at = now()
 WHERE concept_code IN ('doctype.dpa', 'doctype.side_letter',
                        'doctype.renewal', 'doctype.guaranty')
   AND (status = 'proposed' OR pipeline_doc_type IS DISTINCT FROM 'contract');

COMMIT;
