-- Prompt engine, phase 1, milestone 2: the registry.
-- Design: specs/2026-10-13-prompt-engine-phase1-design.md
--
-- A prompt is a name, a list of versions that never change once saved, and a
-- pointer saying which version is in use. Promotion and rollback move the
-- pointer; nothing is ever edited in place.
--
-- proc.bp_prompt is NOT altered. It stays as the table the older readers use,
-- and a promotion writes the live wording back into it (see
-- services/prompt_engine/registry.py) until those readers are migrated.
--
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb.
-- Rollback: 2026-10-13_bp_prompt_registry_rollback.sql

CREATE SCHEMA IF NOT EXISTS proc;

-- One row per prompt or fragment. The key is what everything refers to: a
-- prompt_id differs between databases, a key does not.
CREATE TABLE IF NOT EXISTS proc.bp_prompt_registry (
    prompt_key   TEXT PRIMARY KEY CHECK (prompt_key ~ '^[a-z0-9][a-z0-9_.-]*$'),
    kind         TEXT NOT NULL DEFAULT 'prompt' CHECK (kind IN ('prompt', 'fragment')),
    owner        TEXT,
    created_by   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION proc.bp_prompt_registry_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'a prompt is retired by moving its pointer, never deleted';
    END IF;
    IF NEW.prompt_key IS DISTINCT FROM OLD.prompt_key OR NEW.kind IS DISTINCT FROM OLD.kind THEN
        RAISE EXCEPTION 'a prompt''s key and kind cannot be changed';
    END IF;
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS tr_bp_prompt_registry_guard ON proc.bp_prompt_registry;
CREATE TRIGGER tr_bp_prompt_registry_guard
    BEFORE UPDATE OR DELETE ON proc.bp_prompt_registry
    FOR EACH ROW EXECUTE FUNCTION proc.bp_prompt_registry_guard();

-- Every save is a row here, and a row here is never changed.
CREATE TABLE IF NOT EXISTS proc.bp_prompt_version (
    prompt_key      TEXT NOT NULL REFERENCES proc.bp_prompt_registry (prompt_key),
    version         INTEGER NOT NULL CHECK (version >= 1),
    -- 'legacy': imported as it was and served as it was. 'v1': typed template.
    syntax          TEXT NOT NULL CHECK (syntax IN ('legacy', 'v1')),
    template        TEXT NOT NULL,
    variables       JSONB NOT NULL DEFAULT '[]',
    -- Every fragment this version uses, as [{"key": ..., "version": ...}].
    fragment_refs   JSONB NOT NULL DEFAULT '[]',
    -- Output schema, examples, default parameters, tested models.
    settings        JSONB NOT NULL DEFAULT '{}',
    change_note     TEXT,
    author          TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_sha256  TEXT NOT NULL,
    PRIMARY KEY (prompt_key, version)
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_version_fragment_refs
    ON proc.bp_prompt_version USING GIN (fragment_refs jsonb_path_ops);

CREATE OR REPLACE FUNCTION proc.bp_prompt_version_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_prompt_version rows cannot be changed once saved (%)', TG_OP;
END $$;
DROP TRIGGER IF EXISTS tr_bp_prompt_version_immutable ON proc.bp_prompt_version;
CREATE TRIGGER tr_bp_prompt_version_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_prompt_version
    FOR EACH ROW EXECUTE FUNCTION proc.bp_prompt_version_immutable();

-- The pointer, and its whole history, and the approval record, in one table:
-- every promotion and rollback is a row, and the pointer is the latest row.
CREATE TABLE IF NOT EXISTS proc.bp_prompt_pointer_move (
    move_id       BIGSERIAL PRIMARY KEY,
    prompt_key    TEXT NOT NULL,
    pointer       TEXT NOT NULL DEFAULT 'live' CHECK (pointer ~ '^[a-z][a-z0-9_-]*$'),
    from_version  INTEGER,
    to_version    INTEGER NOT NULL,
    change        TEXT NOT NULL CHECK (change IN ('promoted', 'rolled_back', 'imported')),
    approved_by   TEXT NOT NULL,
    note          TEXT,
    moved_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (prompt_key, to_version)
        REFERENCES proc.bp_prompt_version (prompt_key, version)
);
CREATE INDEX IF NOT EXISTS ix_bp_prompt_pointer_move_latest
    ON proc.bp_prompt_pointer_move (prompt_key, pointer, move_id DESC);

CREATE OR REPLACE FUNCTION proc.bp_prompt_pointer_move_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_prompt_pointer_move is append-only (%)', TG_OP;
END $$;
DROP TRIGGER IF EXISTS tr_bp_prompt_pointer_move_immutable ON proc.bp_prompt_pointer_move;
CREATE TRIGGER tr_bp_prompt_pointer_move_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_prompt_pointer_move
    FOR EACH ROW EXECUTE FUNCTION proc.bp_prompt_pointer_move_immutable();
