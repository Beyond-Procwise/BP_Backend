-- Agent policy governance, stage 5 (learning). Additive and idempotent.
BEGIN;

-- One row per (policy, change). policy_key has no foreign key, like bp_policy_firing: the scan
-- must be able to keep a row about a policy whatever happens to it, and tests use keys that
-- are not real policies.
CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_suggestion (
    suggestion_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    policy_key     TEXT NOT NULL,
    change_key     TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('threshold','list')),
    proposal       JSONB NOT NULL,
    stats          JSONB NOT NULL,
    confident      BOOLEAN NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open','not_yet','dismissed','saved','superseded')),
    n_at_action    INTEGER,
    acted_by       TEXT,
    acted_at       TIMESTAMPTZ,
    saved_version  INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_bp_agent_policy_suggestion_change UNIQUE (policy_key, change_key)
);
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_suggestion_status
    ON proc.bp_agent_policy_suggestion (status, policy_key);
COMMENT ON TABLE proc.bp_agent_policy_suggestion IS
    'A change learning suggests to a policy owner. Never applied by code: only a person''s save changes a policy.';

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_second_review (
    policy_key    TEXT NOT NULL,
    version       INTEGER NOT NULL,
    suggestion_id BIGINT,
    reviewed_by   TEXT NOT NULL,
    reviewed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    note          TEXT,
    PRIMARY KEY (policy_key, version)
);
COMMENT ON TABLE proc.bp_agent_policy_second_review IS
    'A second person reviewed this saved draft version of a policy that loosens it as learning suggested.';

-- A person's record that an approved action was later undone (a refund clawed back).
CREATE TABLE IF NOT EXISTS proc.bp_policy_firing_reversal (
    firing_id   BIGINT PRIMARY KEY REFERENCES proc.bp_policy_firing (firing_id),
    reversed_by TEXT NOT NULL,
    reversed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason      TEXT NOT NULL CHECK (btrim(reason) <> '')
);

CREATE OR REPLACE FUNCTION proc.bp_policy_firing_reversal_guard() RETURNS trigger AS $fn$
BEGIN
    RAISE EXCEPTION 'bp_policy_firing_reversal is append-only: % refused', TG_OP;
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_bp_policy_firing_reversal_guard ON proc.bp_policy_firing_reversal;
CREATE TRIGGER trg_bp_policy_firing_reversal_guard
    BEFORE UPDATE OR DELETE ON proc.bp_policy_firing_reversal
    FOR EACH ROW EXECUTE FUNCTION proc.bp_policy_firing_reversal_guard();

COMMIT;
