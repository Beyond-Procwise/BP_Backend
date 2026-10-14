-- Prompt engine, phase 1, milestone 4: the manifest store.
-- Design: specs/2026-10-13-prompt-engine-phase1-design.md
--
-- bp_prompt_render      what an agent was given, stored ONCE per distinct text,
--                       under the sha256 of that text. Holds prompt wording with
--                       customer values filled in, so it is the restricted
--                       table: read only through services/prompt_engine/render.py.
--                       The TEXT (`rendered`) is removed once it has gone unused
--                       for PromptEnginePolicy.manifest_retention_days. The
--                       manifest stays: which prompt version, which fragment
--                       versions, which obligations, how many tokens. It holds
--                       no wording and no values (owner ruling 2026-10-13: the
--                       text goes after 7 days, the metadata is kept).
-- bp_prompt_render_use  one row per render call: which run and step used which
--                       hash. No wording, no values. Append-only and kept.
--
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb.
-- Rollback: 2026-10-13_bp_prompt_render_rollback.sql

CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_prompt_render (
    render_hash        TEXT PRIMARY KEY CHECK (render_hash ~ '^sha256:[0-9a-f]{64}$'),
    prompt_key         TEXT NOT NULL,
    version            INTEGER NOT NULL,
    manifest           JSONB NOT NULL,
    rendered           JSONB,               -- NULL once the text has been removed
    first_rendered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    text_removed_at    TIMESTAMPTZ,
    CONSTRAINT ck_bp_prompt_render_text CHECK ((rendered IS NULL) = (text_removed_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_render_last_used
    ON proc.bp_prompt_render (last_used_at) WHERE rendered IS NOT NULL;

CREATE TABLE IF NOT EXISTS proc.bp_prompt_render_use (
    use_id       BIGSERIAL PRIMARY KEY,
    render_hash  TEXT NOT NULL,
    prompt_key   TEXT NOT NULL,
    version      INTEGER NOT NULL,
    run_id       TEXT,
    step_id      TEXT,
    attempt      INTEGER,
    decision_id  TEXT,
    used_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_render_use_run
    ON proc.bp_prompt_render_use (run_id, use_id);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_render_use_hash
    ON proc.bp_prompt_render_use (render_hash);

CREATE OR REPLACE FUNCTION proc.bp_prompt_render_use_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_prompt_render_use is append-only (%)', TG_OP;
END $$;
DROP TRIGGER IF EXISTS tr_bp_prompt_render_use_immutable ON proc.bp_prompt_render_use;
CREATE TRIGGER tr_bp_prompt_render_use_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_prompt_render_use
    FOR EACH ROW EXECUTE FUNCTION proc.bp_prompt_render_use_immutable();
