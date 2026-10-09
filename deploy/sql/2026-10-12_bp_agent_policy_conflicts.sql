-- 2026-10-12  Agent policy governance, stage 4: conflicts. Additive, idempotent.
-- Cases themselves are proc.bp_decision rows (subject_type policy_conflict | live_conflict);
-- this table indexes them by policy so a pair has at most one open policy case and a
-- policy's history can list its conflicts. proc.bp_policy is untouched (ruling B).
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_conflict (
    decision_id     BIGINT PRIMARY KEY REFERENCES proc.bp_decision (decision_id),
    kind            TEXT NOT NULL CHECK (kind IN ('policy','live')),
    pair_key        TEXT NOT NULL,                 -- sorted policy keys joined by '|'
    policy_keys     TEXT[] NOT NULL CHECK (cardinality(policy_keys) >= 2),
    policy_versions JSONB NOT NULL,                -- {"FIN-0012": 3, "CUS-0004": 1}
    raised_by       TEXT NOT NULL CHECK (raised_by IN ('save','scan','live','repeat')),
    is_open         BOOLEAN NOT NULL DEFAULT TRUE,
    outcome         TEXT,                          -- the returned decision value
    decided_by      TEXT,
    decided_at      TIMESTAMPTZ,
    by_person       BOOLEAN,                       -- false for timeouts, block records, standing rules
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_agent_policy_conflict_settled
        CHECK (is_open OR (outcome IS NOT NULL AND decided_at IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_open_pair
    ON proc.bp_agent_policy_conflict (pair_key) WHERE kind = 'policy' AND is_open;
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_keys
    ON proc.bp_agent_policy_conflict USING GIN (policy_keys);
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_pair_decided
    ON proc.bp_agent_policy_conflict (kind, pair_key, decided_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_conflict_rule (
    rule_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pair_key       TEXT NOT NULL,
    prevails       TEXT NOT NULL,
    yields         TEXT NOT NULL,
    rule_text      TEXT NOT NULL,                  -- "FIN-0012 takes priority over CUS-0004"
    decision_id    BIGINT NOT NULL REFERENCES proc.bp_decision (decision_id),
    decided_by     TEXT NOT NULL,
    decided_at     TIMESTAMPTZ NOT NULL,
    superseded_at  TIMESTAMPTZ,
    superseded_by  BIGINT REFERENCES proc.bp_decision (decision_id),
    CONSTRAINT ck_bp_agent_policy_conflict_rule_two CHECK (prevails <> yields)
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_agent_policy_conflict_rule_in_force
    ON proc.bp_agent_policy_conflict_rule (pair_key) WHERE superseded_at IS NULL;

-- Policy-case notifications (to owners) have no firing row: the firing log is for actions.
ALTER TABLE proc.bp_policy_notification ALTER COLUMN firing_id DROP NOT NULL;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_bp_policy_notification_target') THEN
    ALTER TABLE proc.bp_policy_notification ADD CONSTRAINT ck_bp_policy_notification_target
      CHECK (firing_id IS NOT NULL OR link LIKE 'conflict:%');
  END IF;
END $$;
COMMIT;
