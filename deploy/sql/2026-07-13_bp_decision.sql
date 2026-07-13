-- proc.bp_decision — every decision the system makes, and the facts behind it.
--
-- The point of this table is TRACEABILITY. A decision you cannot check is a guess
-- with better PR. Each row records not just the verdict but the evidence it was
-- computed from (`facts`), the governed rule that was applied (`policy_*`), and
-- whether the engine could resolve it at all (`resolution`).
--
-- `facts` is the load-bearing column. If it is empty, the decision was not grounded
-- and must not have been made automatically.

CREATE TABLE IF NOT EXISTS proc.bp_decision (
    decision_id      BIGSERIAL PRIMARY KEY,

    -- What was being decided. subject_type is the kind of thing ('finding',
    -- 'approval', 'quote_award', 'discrepancy'); subject_id its key in that domain.
    subject_type     TEXT NOT NULL,
    subject_id       TEXT,
    deal_id          TEXT,
    supplier_id      TEXT,

    -- The verdict, and how confident the engine was that it could make it at all.
    --   resolution: 'resolved'  -> policy + facts gave a clear answer
    --               'escalated' -> the engine could NOT resolve it; a human must
    --   decision:   the chosen option (e.g. approve / reject / hold / investigate)
    decision         TEXT NOT NULL,
    resolution       TEXT NOT NULL DEFAULT 'resolved',
    rationale        TEXT,

    -- Provenance. Which governed rule produced this, and WHAT WAS TRUE when it did.
    policy_id        BIGINT,
    policy_name      TEXT,
    facts            JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Where each fact came from: table/column/row, or the tool that returned it.
    -- This is what lets someone re-derive the decision from source.
    evidence         JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Human-in-the-loop.
    status           TEXT NOT NULL DEFAULT 'open',   -- open | actioned | overridden
    actioned_by      TEXT,
    actioned_at      TIMESTAMPTZ,
    override_reason  TEXT,

    workflow_id      TEXT,
    agent            TEXT,
    created_by       TEXT NOT NULL DEFAULT 'system',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_bp_decision_subject    ON proc.bp_decision (subject_type, subject_id);
CREATE INDEX IF NOT EXISTS ix_bp_decision_deal_id    ON proc.bp_decision (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_decision_resolution ON proc.bp_decision (resolution);
CREATE INDEX IF NOT EXISTS ix_bp_decision_status     ON proc.bp_decision (status);
CREATE INDEX IF NOT EXISTS ix_bp_decision_created    ON proc.bp_decision (created_at DESC);
