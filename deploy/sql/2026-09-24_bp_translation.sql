-- 2026-09-24  AI translation: cached translations and per-user language preference
-- ---------------------------------------------------------------------------
-- proc.bp_translation holds one translated string per (language, English source hash,
-- prompt version, model). English itself is never stored as a translation; it is the
-- source. origin='reviewed' rows are human translations (imported from the UI's former
-- es/fr/de dictionaries) and are served ahead of any machine row, whatever the prompt
-- version or model. prompt_version/model on reviewed rows are the literals
-- 'reviewed'/'human'.
--
-- proc.bp_user_preference holds a signed-in user's settings by key (first key:
-- 'language', value {"code": "...", "name": "..."}).
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_translation (
    target_lang     TEXT        NOT NULL,
    source_hash     TEXT        NOT NULL,
    prompt_version  TEXT        NOT NULL,
    model           TEXT        NOT NULL,
    source_text     TEXT        NOT NULL,
    translated_text TEXT        NOT NULL,
    origin          TEXT        NOT NULL CHECK (origin IN ('machine', 'reviewed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target_lang, source_hash, prompt_version, model)
);
CREATE INDEX IF NOT EXISTS ix_bp_translation_lang_hash
    ON proc.bp_translation (target_lang, source_hash);

CREATE TABLE IF NOT EXISTS proc.bp_user_preference (
    user_subject TEXT        NOT NULL,
    pref_key     TEXT        NOT NULL,
    pref_value   JSONB       NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_subject, pref_key)
);

COMMIT;
