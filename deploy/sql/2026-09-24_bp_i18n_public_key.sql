-- 2026-09-24  AI translation: the UI strings a signed-out visitor may read translated
-- ---------------------------------------------------------------------------
-- The sign-in, password-reset and landing screens are shown before anyone is
-- identified, so they are served by the one public translation path,
-- GET /i18n/public/{lang}. That path accepts no text and never calls the model:
-- it returns cached translations of exactly the keys listed here.
-- The list is data, published from the UI's English catalog by
--   scripts/i18n_pretranslate.py --publish-public
-- (key prefixes in config/i18n/public.json). Replacing the list is audited.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_i18n_public_key (
    msg_key     TEXT        PRIMARY KEY,
    source_text TEXT        NOT NULL,
    source_hash TEXT        NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
