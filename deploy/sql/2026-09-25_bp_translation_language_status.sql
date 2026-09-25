-- 2026-09-25  AI translation: what the model said about each target language
-- ---------------------------------------------------------------------------
-- The prompt (translate-ui v3) asks the model, per batch, whether it recognises the target
-- language (lang_recognized) and how confident it is (confidence: high/medium/low). The
-- verdict is kept per (language, prompt version, model) so every user and the background
-- queue see it:
--   recognized = false  -> the language is served in English and not sent again until the
--                          prompt version or model changes (sticky: one "no" is enough).
--   confidence          -> the LOWEST seen; 'low' marks the language Experimental.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_translation_language_status (
    target_lang    TEXT        NOT NULL,
    prompt_version TEXT        NOT NULL,
    model          TEXT        NOT NULL,
    recognized     BOOLEAN,
    confidence     TEXT        CHECK (confidence IN ('high', 'medium', 'low')),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (target_lang, prompt_version, model)
);

COMMIT;
