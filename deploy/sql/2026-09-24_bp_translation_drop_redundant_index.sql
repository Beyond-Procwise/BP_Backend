-- 2026-09-24  AI translation: drop an index the primary key already provides
-- ---------------------------------------------------------------------------
-- ix_bp_translation_lang_hash covered (target_lang, source_hash), which are the leading
-- columns of the primary key (target_lang, source_hash, prompt_version, model). Every
-- lookup it served is served by the key; it only cost write time.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_translation_lang_hash;
COMMIT;
