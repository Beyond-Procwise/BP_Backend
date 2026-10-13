-- Prompt engine, phase 1, milestone 4: the manifest store.
-- Design: specs/2026-10-13-prompt-engine-phase1-design.md
--
-- bp_prompt_render      what an agent was given, stored ONCE per distinct text,
--                       under the sha256 of that text. Holds prompt wording with
--                       customer values filled in, so it is the restricted
--                       table: read only through services/prompt_engine/render.py.
--                       Aged out after PromptEnginePolicy.manifest_retention_days
--                       from the last time it was used.
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
    rendered           JSONB NOT NULL,
    first_rendered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_render_last_used
    ON proc.bp_prompt_render (last_used_at);

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
