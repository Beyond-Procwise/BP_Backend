-- Agent policy governance, stage 3 (enforcement). Additive and idempotent.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_policy_decider_map (
    decider_name  TEXT PRIMARY KEY,
    groups        TEXT[] NOT NULL DEFAULT '{}',
    emails        TEXT[] NOT NULL DEFAULT '{}',
    notes         TEXT,
    last_modified_by TEXT NOT NULL,
    last_modified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_policy_decider_map_someone CHECK (cardinality(groups) + cardinality(emails) > 0)
);

CREATE TABLE IF NOT EXISTS proc.bp_policy_firing (
    firing_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_key    TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    checkpoint    TEXT NOT NULL,
    action_name   TEXT NOT NULL,
    agent         TEXT,
    workflow_id   TEXT,
    requested_by  TEXT,
    outcome       TEXT NOT NULL CHECK (outcome IN ('approve','block','notify')),
    result        TEXT NOT NULL CHECK (result IN ('allowed','paused_for_approval','approved','rejected','blocked','timed_out','error')),
    matched_values JSONB NOT NULL DEFAULT '{}',
    missing_inputs TEXT[] NOT NULL DEFAULT '{}',
    decision_id   BIGINT,
    decided_level INTEGER,
    decided_by    TEXT,
    decided_at    TIMESTAMPTZ,
    reason        TEXT,
    duration_ms   INTEGER,
    reversal_of   BIGINT REFERENCES proc.bp_policy_firing (firing_id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_firing_policy ON proc.bp_policy_firing (policy_key, created_at);
CREATE INDEX IF NOT EXISTS ix_bp_policy_firing_decision ON proc.bp_policy_firing (decision_id);

-- Append-only: DELETE is refused; an UPDATE is allowed only on a row whose result is
-- 'paused_for_approval' and only to the decision columns.
CREATE OR REPLACE FUNCTION proc.bp_policy_firing_guard() RETURNS trigger AS $fn$
DECLARE
    decision_cols CONSTANT text[] := ARRAY['decision_id','decided_level','decided_by','decided_at','reason','result'];
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'bp_policy_firing is append-only: DELETE refused';
    END IF;
    IF OLD.result IS DISTINCT FROM 'paused_for_approval' THEN
        RAISE EXCEPTION 'bp_policy_firing is append-only: only a paused_for_approval row may be updated';
    END IF;
    IF (to_jsonb(NEW) - decision_cols) IS DISTINCT FROM (to_jsonb(OLD) - decision_cols) THEN
        RAISE EXCEPTION 'bp_policy_firing is append-only: only decision columns may change';
    END IF;
    RETURN NEW;
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_bp_policy_firing_guard ON proc.bp_policy_firing;
CREATE TRIGGER trg_bp_policy_firing_guard
    BEFORE UPDATE OR DELETE ON proc.bp_policy_firing
    FOR EACH ROW EXECUTE FUNCTION proc.bp_policy_firing_guard();

CREATE TABLE IF NOT EXISTS proc.bp_policy_notification (
    notification_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    firing_id     BIGINT NOT NULL REFERENCES proc.bp_policy_firing (firing_id),
    recipient     TEXT NOT NULL,
    message       TEXT NOT NULL,
    link          TEXT NOT NULL,
    read_by       TEXT[] NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_notification_recipient ON proc.bp_policy_notification (recipient, created_at);

ALTER TABLE proc.bp_decision
    ADD COLUMN IF NOT EXISTS options       JSONB,
    ADD COLUMN IF NOT EXISTS respond_by    TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS on_timeout    TEXT,
    ADD COLUMN IF NOT EXISTS decision_scope TEXT,
    ADD COLUMN IF NOT EXISTS levels        JSONB,
    ADD COLUMN IF NOT EXISTS current_level INTEGER;
CREATE INDEX IF NOT EXISTS ix_bp_decision_open_respond_by ON proc.bp_decision (respond_by) WHERE status = 'open';
COMMIT;
