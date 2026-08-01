-- 2026-08-01  Analysis events
-- ---------------------------------------------------------------------------
-- An analysis becomes a first-class record instead of being indistinguishable
-- from a draft deal (proc.bp_deal.is_tracked = false). One row per analysis
-- RUN: a name, a date, the documents it read, what it found, and the deals it
-- produced.
--
-- version lives on bp_analysis_deal, NOT on bp_analysis: one run can touch
-- several deals, and each of those deals is at a different point in its own
-- history, so the same run may be v3 for one deal and v1 for another.
--
-- Idempotent. Run against: bp_sqldb.
-- See docs/superpowers/specs/2026-08-01-analysis-events-design.md
-- ---------------------------------------------------------------------------

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_analysis (
    analysis_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT,
    mode            TEXT NOT NULL DEFAULT 'new'
                    CHECK (mode IN ('new', 'amend', 'bulk')),
    -- UNIQUE so creating an event is idempotent: one upload can never produce
    -- two events, whether it is created by the UI or by the sweep.
    session_id      TEXT UNIQUE,
    status          TEXT NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'complete', 'failed')),
    failure_reason  TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    created_by      TEXT,
    document_count  INTEGER,
    value_found     NUMERIC(18,2),
    currency        VARCHAR(8),
    findings        JSONB
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_started
    ON proc.bp_analysis (started_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_analysis_status
    ON proc.bp_analysis (status) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS proc.bp_analysis_document (
    analysis_id   UUID NOT NULL
                  REFERENCES proc.bp_analysis(analysis_id) ON DELETE CASCADE,
    doc_type      TEXT,
    doc_pk        TEXT,
    file_path     TEXT NOT NULL,
    file_name     TEXT,
    outcome       TEXT CHECK (outcome IN ('target', 'discrepancy', 'failed')),
    PRIMARY KEY (analysis_id, file_path)
);

CREATE TABLE IF NOT EXISTS proc.bp_analysis_deal (
    analysis_id  UUID NOT NULL
                 REFERENCES proc.bp_analysis(analysis_id) ON DELETE CASCADE,
    deal_id      VARCHAR(25) NOT NULL,
    version      INTEGER NOT NULL,
    is_latest    BOOLEAN NOT NULL DEFAULT true,
    linked_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (analysis_id, deal_id),
    UNIQUE (deal_id, version)
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_deal_deal
    ON proc.bp_analysis_deal (deal_id, version DESC);

COMMENT ON TABLE proc.bp_analysis IS
    'One row per analysis RUN. Frozen at session resolution: findings is a '
    'point-in-time snapshot and is never updated afterwards.';
COMMENT ON COLUMN proc.bp_analysis_deal.version IS
    'The nth analysis for THIS deal. Allocated per deal_id, not globally.';

COMMIT;
