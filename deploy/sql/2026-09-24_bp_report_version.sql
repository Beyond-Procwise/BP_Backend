-- 2026-09-24  Report versions: a light editor over released reports.
-- ---------------------------------------------------------------------------
-- Ruled by the user 2026-09-24: Buyers and above may edit a released report's words and
-- layout -- never its numbers -- and every saved edit is a new version that resets sign-off.
--
-- An edit is re-rendered from the report's STORED Fact Pack (fact_pack), never re-queried,
-- so the figures stay exactly the verified ones. Version 1 is the agent's release; each save
-- adds the next. bp_report_job's deck/page are always the current version's files, so the
-- download and sign-off paths are unchanged. Jobs released before this have no fact_pack and
-- cannot be edited.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

ALTER TABLE proc.bp_report_job
    ADD COLUMN IF NOT EXISTS fact_pack       JSONB,
    ADD COLUMN IF NOT EXISTS title           TEXT,
    ADD COLUMN IF NOT EXISTS current_version INTEGER,
    ADD COLUMN IF NOT EXISTS last_edited_by  TEXT;

CREATE TABLE IF NOT EXISTS proc.bp_report_version (
    job_id       TEXT    NOT NULL REFERENCES proc.bp_report_job (job_id) ON DELETE CASCADE,
    version      INTEGER NOT NULL CHECK (version >= 1),
    title        TEXT    NOT NULL,
    ast          JSONB   NOT NULL,
    deck         BYTEA   NOT NULL,
    page         BYTEA,
    deck_sha256  TEXT    NOT NULL,
    page_sha256  TEXT,
    edited_by    TEXT,              -- NULL for version 1: the agent wrote it
    edited_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    summary      TEXT,
    PRIMARY KEY (job_id, version)
);

COMMIT;
