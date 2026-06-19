-- Structured per-deal analytics row backing the UI "Detailed Analysis Summary"
-- grid. One current row per deal (is_current pattern, mirrors proc.bp_summary).
-- All values are computed deterministically from the deal's _trgt documents;
-- anything not derivable stays NULL (no fabrication).

CREATE TABLE IF NOT EXISTS proc.bp_analysis_summary (
    analysis_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id              VARCHAR(25) NOT NULL,
    deal_name            VARCHAR,
    supplier             TEXT,
    category             TEXT,
    deal_value           NUMERIC(18,2),
    currency             VARCHAR(8),
    volume               NUMERIC(18,2),
    unit_price           NUMERIC(18,4),
    price_change_pct     NUMERIC(9,2),
    volume_change_pct    NUMERIC(9,2),
    efficiency_score     NUMERIC(18,2),
    items                JSONB,
    item_count           INTEGER,
    narrative_summary_id UUID,
    data_snapshot        JSONB,
    model                TEXT,
    is_current           BOOLEAN     NOT NULL DEFAULT true,
    generated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_analysis_summary_current
    ON proc.bp_analysis_summary (deal_id) WHERE is_current;
CREATE INDEX IF NOT EXISTS ix_bp_analysis_summary_deal
    ON proc.bp_analysis_summary (deal_id, generated_at DESC);
