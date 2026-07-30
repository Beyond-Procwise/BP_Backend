-- deploy/sql/2026-07-29_bp_negotiation_advice.sql
-- Buyer-facing negotiation advice: one row per advice session, plus the
-- buyer-stated facts that shaped it. Stated facts are kept apart from measured
-- signals by construction so a stated value can never become data.
-- Additive + idempotent.
--
-- bp_policy column list verified against the live schema before writing this
-- (the plan says not to guess it): policy_id is GENERATED ALWAYS AS IDENTITY so
-- it must be omitted, and policy_status / version / created_* / last_modified_*
-- all carry defaults, so the four columns below are the complete set required.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_negotiation_advice (
    advice_id            VARCHAR PRIMARY KEY,
    deal_id              VARCHAR NOT NULL,
    supplier_id          VARCHAR,
    quadrant             VARCHAR,
    quadrant_source      VARCHAR NOT NULL DEFAULT 'computed',
    quadrant_confidence  NUMERIC,
    style                VARCHAR,
    style_source         VARCHAR NOT NULL DEFAULT 'computed',
    signals              JSONB   DEFAULT '{}'::jsonb,
    plays                JSONB   DEFAULT '[]'::jsonb,
    created_by           VARCHAR,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT bp_negotiation_advice_quadrant_source_check
        CHECK (quadrant_source IN ('computed', 'buyer')),
    CONSTRAINT bp_negotiation_advice_style_source_check
        CHECK (style_source IN ('computed', 'buyer'))
);

CREATE INDEX IF NOT EXISTS ix_bp_negotiation_advice_deal_id
    ON proc.bp_negotiation_advice (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_negotiation_advice_created_at
    ON proc.bp_negotiation_advice (created_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_negotiation_advice_fact (
    advice_id     VARCHAR NOT NULL,
    fact_key      VARCHAR NOT NULL,
    fact_value    TEXT,
    stated_by     VARCHAR,
    stated_at     TIMESTAMPTZ DEFAULT NOW(),
    withdrawn_at  TIMESTAMPTZ,
    PRIMARY KEY (advice_id, fact_key)
);

CREATE INDEX IF NOT EXISTS ix_bp_negotiation_advice_fact_advice_id
    ON proc.bp_negotiation_advice_fact (advice_id);

-- Governed thresholds. Seeded from the live distribution: deal-value p90
-- (98175; median 4180) and the per-DEAL median alternative-supplier count (93,
-- measured over 80 sampled deals; the per-item median is far lower and would
-- make every deal "many alternatives"). Testdata-derived —
-- retune against real spend, which is why these are data and not constants.
INSERT INTO proc.bp_policy (policy_type, policy_name, policy_details, policy_status)
SELECT 'negotiation', 'negotiation_advice_thresholds',
       '{"high_spend": 98175.0, "many_alternatives": 93}'::jsonb, 1
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
    WHERE policy_name = 'negotiation_advice_thresholds'
);

COMMIT;
