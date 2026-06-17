-- 2026-06-17 Upstream procurement requirement entity ("Need Identified" stage).
-- Gathered by RequirementsAgent via conversation; precedes any deal_id.
-- Additive + idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_requirement (
    requirement_id      VARCHAR PRIMARY KEY,
    session_id          VARCHAR,
    status              VARCHAR NOT NULL DEFAULT 'gathering',
    created_by          VARCHAR,
    title               TEXT,
    category            VARCHAR,
    description         TEXT,
    quantity            NUMERIC,
    unit                VARCHAR,
    target_budget       NUMERIC,
    currency            VARCHAR,
    needed_by_date      DATE,
    delivery_location   TEXT,
    priority            VARCHAR,
    specifications      JSONB DEFAULT '{}'::jsonb,
    constraints         JSONB DEFAULT '{}'::jsonb,
    completeness_score  NUMERIC DEFAULT 0,
    missing_fields      JSONB DEFAULT '[]'::jsonb,
    seed_context        JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT bp_requirement_status_check CHECK (
        status IN ('draft', 'gathering', 'complete', 'handed_off', 'abandoned'))
);

CREATE INDEX IF NOT EXISTS ix_bp_requirement_status      ON proc.bp_requirement (status);
CREATE INDEX IF NOT EXISTS ix_bp_requirement_category    ON proc.bp_requirement (category);
CREATE INDEX IF NOT EXISTS ix_bp_requirement_created_by  ON proc.bp_requirement (created_by);

COMMIT;
