-- Created agents move out of agent_definitions.json (a git-tracked file that a
-- redeploy replaces) into the database, with who made each one and a row for
-- every change. The built-in agents stay in the file.
--
-- The server creates these itself at startup (repositories/agent_definition_repo.py,
-- same DDL) and takes in any created agent still sitting in the file. This file
-- is for applying the tables ahead of a deploy. Additive; safe to run twice.

CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_definition (
    slug          TEXT PRIMARY KEY,
    definition    JSONB NOT NULL,
    derived_from  TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    status        SMALLINT NOT NULL DEFAULT 1,      -- 1 live, 0 deleted
    created_by    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by    TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_definition_status
    ON proc.bp_agent_definition (status);

CREATE TABLE IF NOT EXISTS proc.bp_agent_definition_version (
    slug        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    change      TEXT NOT NULL
                CHECK (change IN ('created', 'updated', 'published', 'deleted', 'imported')),
    definition  JSONB NOT NULL,
    changed_by  TEXT,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (slug, version)
);
