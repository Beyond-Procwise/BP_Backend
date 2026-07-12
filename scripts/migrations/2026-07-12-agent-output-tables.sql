-- Persist the output of SupplierRankingAgent and QuoteEvaluationAgent.
--
-- Neither agent persisted anything. Their results existed only in the workflow blackboard
-- and the HTTP response, and vanished when the run ended — so the Suppliers view shows
-- master data with no ranking, and the Quotes view hard-nulls the columns these agents
-- could fill. proc.procurement_flow (0 rows) and proc.supplier_responses (never created)
-- were the tables an earlier design intended; QuoteEvaluationAgent still READS
-- supplier_responses and carries a "TODO: no bp_supplier_responses table yet".
--
-- bp_ prefix per the project convention; ix_bp_<table>_<col> for indexes.

CREATE TABLE IF NOT EXISTS proc.bp_supplier_ranking (
    ranking_id          BIGSERIAL PRIMARY KEY,
    workflow_id         TEXT,
    supplier_id         TEXT NOT NULL,
    supplier_name       TEXT,
    rank_position       INTEGER,
    rank_count          INTEGER,
    final_score         NUMERIC(8,2),
    price_score         NUMERIC(8,2),
    delivery_score      NUMERIC(8,2),
    risk_score          NUMERIC(8,2),
    payment_terms_score NUMERIC(8,2),
    avg_unit_price      NUMERIC(18,2),
    total_spend         NUMERIC(18,2),
    po_count            INTEGER,
    invoice_count       INTEGER,
    lead_time_days      NUMERIC(8,2),
    justification       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_supplier_ranking_supplier_id
    ON proc.bp_supplier_ranking (supplier_id);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_ranking_created_at
    ON proc.bp_supplier_ranking (created_at DESC);


CREATE TABLE IF NOT EXISTS proc.bp_quote_evaluation (
    evaluation_id       BIGSERIAL PRIMARY KEY,
    workflow_id         TEXT,
    quote_id            TEXT NOT NULL,
    supplier_id         TEXT,
    deal_id             TEXT,
    currency            TEXT,
    total_amount        NUMERIC(18,2),
    total_line_amount   NUMERIC(18,2),
    avg_unit_price      NUMERIC(18,2),
    line_items_count    INTEGER,
    category_match      BOOLEAN,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_quote_evaluation_quote_id
    ON proc.bp_quote_evaluation (quote_id);
CREATE INDEX IF NOT EXISTS ix_bp_quote_evaluation_created_at
    ON proc.bp_quote_evaluation (created_at DESC);
