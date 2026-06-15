-- 2026-06-15 Persisted, stage-tracked opportunities for the Opportunities page.
-- Mined findings (opportunity_miner) are upserted here; the dashboard aggregates
-- from this table. Additive + idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_opportunity (
    opportunity_id        VARCHAR PRIMARY KEY,
    opportunity_ref_id    VARCHAR,
    detector_type         VARCHAR,
    policy_id             VARCHAR,
    supplier_id           VARCHAR,
    supplier_name         VARCHAR,
    category_id           VARCHAR,
    item_id               TEXT,
    item_description      TEXT,
    financial_impact_gbp  NUMERIC,
    realised_savings_gbp  NUMERIC,
    stage                 VARCHAR NOT NULL DEFAULT 'identified',
    deal_id               VARCHAR,
    ml_priority_score     NUMERIC,
    weightage             NUMERIC,
    calculation_details   JSONB,
    source_records        JSONB,
    detected_on           TIMESTAMPTZ,
    stage_updated_at      TIMESTAMPTZ DEFAULT NOW(),
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT bp_opportunity_stage_check CHECK (
        stage IN ('identified', 'negotiation', 'agreed', 'realised', 'closed', 'rejected'))
);

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_stage        ON proc.bp_opportunity (stage);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_detected_on  ON proc.bp_opportunity (detected_on);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_supplier     ON proc.bp_opportunity (supplier_id);
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_detector     ON proc.bp_opportunity (detector_type);

COMMIT;
