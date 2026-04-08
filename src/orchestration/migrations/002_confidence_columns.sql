-- src/orchestration/migrations/002_confidence_columns.sql
-- Add confidence scoring columns to extraction agent tables
-- Spec: Section 6 of orchestration-rearchitecture-design.md

BEGIN;

-- Add confidence_score and needs_review to all agent extraction tables
ALTER TABLE proc.invoice_agent
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

ALTER TABLE proc.purchase_order_agent
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

ALTER TABLE proc.quote_agent
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

ALTER TABLE proc.contracts
    ADD COLUMN IF NOT EXISTS confidence_score REAL,
    ADD COLUMN IF NOT EXISTS needs_review BOOLEAN DEFAULT FALSE;

-- Index for finding records that need review
CREATE INDEX IF NOT EXISTS ix_invoice_agent_review
    ON proc.invoice_agent (needs_review) WHERE needs_review = TRUE;

CREATE INDEX IF NOT EXISTS ix_purchase_order_agent_review
    ON proc.purchase_order_agent (needs_review) WHERE needs_review = TRUE;

CREATE INDEX IF NOT EXISTS ix_quote_agent_review
    ON proc.quote_agent (needs_review) WHERE needs_review = TRUE;

CREATE INDEX IF NOT EXISTS ix_contracts_review
    ON proc.contracts (needs_review) WHERE needs_review = TRUE;

COMMIT;
