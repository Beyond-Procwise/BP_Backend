-- Restores the two bare-word aliases Task 1 dropped:
--   'framework' on doctype.framework_agreement
--   'notice'    on doctype.notice_general  (keeping 'general notice' beside it)
--
-- Why they were dropped and why they come back: Task 1's guard
-- test_an_alias_never_equals_another_concepts_code compared every alias against
-- EVERY concept's local name, so 'framework' clashed with role.framework and
-- 'notice' with role.notice. But the alias index is built from
-- proc.bp_document_type rows ALONE (src/services/concepts/vocabulary.py), so no
-- role.*, link.*, exec.* or event.* code can ever contest an alias. The guard is
-- now scoped to DOCUMENT_TYPE codes and these two aliases create no ambiguity:
-- doctype.termination_notice claims 'termination notice' and
-- 'notice of termination', not bare 'notice', and nothing else claims
-- 'framework'.
--
-- Consequence to know before running: an alias is also an acceptable UPLOAD
-- CATEGORY (src/services/concepts/routing.py). After this,
--   'framework' -> ('contract', 'doctype.framework_agreement')   -- it routes
--   'notice'    -> doctype.notice_general, whose pipeline_doc_type is NULL, so
--                  routing still REFUSES, now with "no pipeline" rather than
--                  "no document type claims it".
--
-- Additive (append only where absent), idempotent (safe to run twice),
-- reversible (_restored_aliases_rollback.sql removes exactly these two).
-- APPEND ORDER IS PART OF THE DATA: Task 1's full-column drift test compares
-- the aliases array in order against seed.py, where both new words are LAST.
BEGIN;

UPDATE proc.bp_document_type
   SET aliases = array_append(aliases, 'framework')
 WHERE concept_code = 'doctype.framework_agreement'
   AND NOT (aliases @> ARRAY['framework']::text[]);

UPDATE proc.bp_document_type
   SET aliases = array_append(aliases, 'notice')
 WHERE concept_code = 'doctype.notice_general'
   AND NOT (aliases @> ARRAY['notice']::text[]);

COMMIT;
