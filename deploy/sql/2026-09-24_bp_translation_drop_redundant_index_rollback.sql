BEGIN;
CREATE INDEX IF NOT EXISTS ix_bp_translation_lang_hash
    ON proc.bp_translation (target_lang, source_hash);
COMMIT;
