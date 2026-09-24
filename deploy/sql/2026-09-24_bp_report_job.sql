-- 2026-09-24  Report jobs: start a report now, collect it later
-- ---------------------------------------------------------------------------
-- Composing a report takes the local model three to five minutes, too long to
-- hold an HTTP request open. POST /reports/generate now files a job here and
-- answers at once; a single background worker runs the jobs one at a time and
-- writes the outcome back; GET /reports/jobs/{job_id} is the poll target.
--
-- Two rules are held by the table, not just by the code:
--
--   * One ACTIVE job per report, scope and as-of day (ux_bp_report_job_active).
--     A double-click, or two people asking for the same quarter, get the same
--     job instead of two GPU runs.
--   * Only a RELEASED job carries a deck (ck_bp_report_job_deck). A blocked
--     report's rendering is never stored, so it can never be downloaded.
--
-- owner is the id of the process that accepted the job. A queued or running
-- job whose owner is not the live process was stranded by a restart and is
-- healed to failed on first read.
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_report_job (
    job_id         TEXT PRIMARY KEY,
    report_type    TEXT NOT NULL,
    scope          JSONB NOT NULL,
    as_of          DATE NOT NULL,
    dedup_key      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued', 'running', 'released', 'blocked', 'failed')),
    owner          TEXT NOT NULL,
    requested_by   TEXT,
    requested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    run_id         TEXT,            -- the Fact Pack id; trace_id in bp_agent_actions
    stage_reached  TEXT,
    blocking       JSONB,           -- the findings that stopped a blocked report
    error          TEXT,            -- a readable reason for a failed job
    deck           BYTEA,
    media_type     TEXT,
    filename       TEXT,
    CONSTRAINT ck_bp_report_job_deck CHECK ((status = 'released') = (deck IS NOT NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_report_job_active
    ON proc.bp_report_job (dedup_key)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS ix_bp_report_job_requested
    ON proc.bp_report_job (requested_at DESC);

COMMIT;
