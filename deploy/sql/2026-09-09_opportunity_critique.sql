-- The Opportunity Critic's verdict, and the gap register that justifies it.
--
-- Keyed on opportunity_ref_id -- the content-derived identity -- and NOT on
-- the per-run counter the miner assigns while walking candidates. It changes
-- for the SAME finding between runs. Keying on it means every verdict orphans
-- on the next mining pass; opportunity_store.py:24 records the cost of
-- learning that once already.
--
-- Gaps are a table, not JSONB on the critique, because the gap register is the
-- part that converts "we could not decide" into work. One gap -- supplier
-- identity -- blocks hundreds of findings, and you learn that from
-- GROUP BY what_is_missing, owner_hint, not by opening 308 JSON blobs.
--
-- Additive, idempotent, reversible.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_opportunity_critique (
    critique_id        BIGSERIAL PRIMARY KEY,
    opportunity_ref_id TEXT        NOT NULL,
    detector_type      TEXT,
    critiqued_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    verdict            TEXT        NOT NULL,
    confidence         TEXT,

    original_claim     TEXT,
    critic_claim       TEXT,
    negotiator_note    TEXT,

    detector_proposed  NUMERIC,
    critic_addressable NUMERIC,
    currency           TEXT,
    value_basis        TEXT,
    haircuts           JSONB,
    lever              JSONB,
    duplicate_of       TEXT,

    -- Every test, including the ones that passed. "anchor_validity PASS" is the
    -- sentence that defends a finding in the room.
    tests              JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Shadow mode, in the shape services/guardrail.py established.
    would_have_suppressed BOOLEAN  NOT NULL DEFAULT false,
    shadowed              BOOLEAN  NOT NULL DEFAULT false,

    -- What produced this verdict. Without these, tuning a threshold silently
    -- leaves stale verdicts on the page presenting themselves as current.
    prompt_version     INTEGER,
    policy_versions    JSONB,
    formula_versions   JSONB,

    -- Pointer to the agent trace in proc.routing.process_details.
    run_id             TEXT,

    CONSTRAINT ck_bp_opportunity_critique_verdict CHECK (verdict IN ('VALID', 'VALID_REFRAMED', 'INVALID', 'UNASSESSED', 'DUPLICATE')),
    CONSTRAINT ck_bp_opportunity_critique_confidence CHECK (confidence IN ('ASSERTED', 'CORROBORATED', 'UNASSESSED'))
);

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_ref
    ON proc.bp_opportunity_critique (opportunity_ref_id, critiqued_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_critiqued
    ON proc.bp_opportunity_critique (critiqued_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_verdict
    ON proc.bp_opportunity_critique (verdict);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_critique_suppressed
    ON proc.bp_opportunity_critique (would_have_suppressed)
    WHERE would_have_suppressed;

CREATE TABLE IF NOT EXISTS proc.bp_opportunity_gap (
    gap_row_id         BIGSERIAL PRIMARY KEY,
    critique_id        BIGINT      NOT NULL
        REFERENCES proc.bp_opportunity_critique (critique_id) ON DELETE CASCADE,
    opportunity_ref_id TEXT        NOT NULL,
    gap_id             TEXT        NOT NULL,
    test               TEXT,
    gap_type           TEXT        NOT NULL,
    what_is_missing    TEXT        NOT NULL,
    why_it_matters     TEXT,
    blocking           BOOLEAN     NOT NULL DEFAULT false,
    resolves_to        TEXT,
    likely_source      TEXT,
    owner_hint         TEXT,
    effort             TEXT,
    ordinal            INTEGER     NOT NULL DEFAULT 0,

    CONSTRAINT ck_bp_opportunity_gap_type CHECK (gap_type IN ('MISSING_EVIDENCE', 'STALE_EVIDENCE', 'UNVERIFIED_ASSERTION', 'NORMALISATION_NEEDED', 'NO_LEVER', 'NO_THRESHOLD', 'DETECTOR_LOGIC')),
    CONSTRAINT ck_bp_opportunity_gap_effort CHECK (effort IN ('LOW', 'MEDIUM', 'HIGH'))
);

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_critique
    ON proc.bp_opportunity_gap (critique_id);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_ref
    ON proc.bp_opportunity_gap (opportunity_ref_id);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_triage
    ON proc.bp_opportunity_gap (blocking, effort);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_gap_owner
    ON proc.bp_opportunity_gap (owner_hint);

COMMIT;
