-- 2026-09-29  Playbook layer (conformance P4).
--
-- A playbook is human-authored expert strategy: when a finding like this
-- appears, run the graph an expert already drew. It is PROPOSED, never run.
-- The selector queues it; a person approves; only then does it execute.
--
-- Nothing is written to proc.bp_agent_workflow -- a playbook references it.
--
-- Additive and idempotent. Safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_playbook (
    playbook_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    playbook_name     TEXT        NOT NULL,
    description       TEXT,
    -- Which finding store this playbook watches.
    trigger_source    TEXT        NOT NULL
        CHECK (trigger_source IN ('detection_finding', 'opportunity')),
    -- Equality match against that store's own columns. Keys are validated at
    -- write time by services.playbooks.finding_source.validate_trigger_match;
    -- an unknown key is refused rather than stored, because a key that matches
    -- nothing is a playbook that silently never fires.
    --   detection_finding: rule_id, category, severity, doc_type, blocks_promotion
    --   opportunity:       detector_type, supplier_id, category_id
    trigger_match     JSONB       NOT NULL DEFAULT '{}',
    -- The expert's strategy: a graph already drawn and saved.
    agent_workflow_id BIGINT      NOT NULL REFERENCES proc.bp_agent_workflow (workflow_id),
    -- Extra static inputs merged into the run payload on execution.
    params            JSONB       NOT NULL DEFAULT '{}',
    playbook_status   TEXT        NOT NULL DEFAULT 'draft'
        CHECK (playbook_status IN ('draft','pending_approval','active','retired')),
    version           INTEGER     NOT NULL DEFAULT 1,
    authored_by       TEXT        NOT NULL,
    approved_by       TEXT,
    approved_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_by  TEXT        NOT NULL DEFAULT 'system',
    -- Active means approved. The two cannot drift apart.
    CONSTRAINT ck_bp_playbook_active_is_approved
        CHECK (playbook_status <> 'active' OR approved_by IS NOT NULL)
);

COMMENT ON TABLE proc.bp_playbook IS
    'Human-authored strategy. Selected for a finding and PROPOSED to a person; never auto-run.';
COMMENT ON COLUMN proc.bp_playbook.trigger_match IS
    'Equality match on the trigger_source store''s own columns. Empty means catch-all, and loses to any more specific playbook.';

CREATE INDEX IF NOT EXISTS ix_bp_playbook_status
    ON proc.bp_playbook (playbook_status);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_source
    ON proc.bp_playbook (trigger_source, playbook_status);

CREATE TABLE IF NOT EXISTS proc.bp_playbook_proposal (
    proposal_id     BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    playbook_id     BIGINT      NOT NULL REFERENCES proc.bp_playbook (playbook_id),
    finding_source  TEXT        NOT NULL
        CHECK (finding_source IN ('detection_finding', 'opportunity')),
    -- TEXT because the two stores disagree on type: bp_detection_finding
    -- .finding_id is BIGINT (stored here as ::text) and bp_opportunity
    -- .opportunity_id is VARCHAR. Deliberately no foreign key -- one column
    -- cannot reference two tables, and finding_source says which.
    finding_id      TEXT        NOT NULL,
    deal_id         TEXT,
    proposal_status TEXT        NOT NULL DEFAULT 'proposed'
        CHECK (proposal_status IN ('proposed','approved','rejected','executed','superseded')),
    -- Which match keys fired, and what the finding's values were. A proposal
    -- must be re-derivable from source, like a decision.
    evidence        JSONB       NOT NULL DEFAULT '{}',
    run_id          TEXT        REFERENCES proc.bp_workflow_run (run_id),
    proposed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by      TEXT,
    decided_at      TIMESTAMPTZ,
    decision_reason TEXT
);

COMMENT ON TABLE proc.bp_playbook_proposal IS
    'One playbook recommended for one finding, awaiting a person. Executed only after approval.';

-- Idempotency. A sweep that runs twice proposes once. Deliberately keyed on
-- playbook_id and NOT on version: editing a playbook must not re-raise a
-- proposal for a finding already proposed.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_playbook_proposal_finding
    ON proc.bp_playbook_proposal (playbook_id, finding_source, finding_id);
CREATE INDEX IF NOT EXISTS ix_bp_playbook_proposal_status
    ON proc.bp_playbook_proposal (proposal_status, proposed_at DESC);

COMMIT;
