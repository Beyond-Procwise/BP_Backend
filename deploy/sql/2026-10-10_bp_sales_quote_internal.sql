-- What a customer quote costs us, held apart from the quote itself (QDA B1).
--
-- Until now cost and margin sat on proc.bp_sales_quote and proc.bp_sales_quote_line beside
-- the customer's own figures, so every read of a quote carried them and only a list of
-- forbidden key names kept them off the customer's page. These two tables hold them instead;
-- the customer read names its columns and has none of these to select.
--
-- STEP 1 OF 2. This creates the tables and copies the existing figures. The seven old
-- columns are left in place, no longer read or written. Dropping them is a separate,
-- separately approved step, after a check that both copies agree.
--
-- Idempotent: re-run it to pick up a quote written to the old columns by a server that had
-- not yet restarted onto the new code.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_sales_quote_internal (
    sales_quote_id  bigint PRIMARY KEY
        REFERENCES proc.bp_sales_quote (sales_quote_id) ON DELETE CASCADE,
    total_cost      numeric(18,2),
    total_margin    numeric(18,2),
    margin_pct      numeric(7,4),
    created_date    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS proc.bp_sales_quote_line_internal (
    sales_quote_line_id  bigint PRIMARY KEY
        REFERENCES proc.bp_sales_quote_line (sales_quote_line_id) ON DELETE CASCADE,
    sales_quote_id       bigint NOT NULL
        REFERENCES proc.bp_sales_quote (sales_quote_id) ON DELETE CASCADE,
    unit_cost            numeric(18,4),      -- snapshot; tier applied at this qty
    cost_tier_applied    numeric,            -- the min_quantity break used, for audit
    line_margin          numeric(18,2),
    line_margin_pct      numeric(7,4),
    created_date         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_sales_quote_line_internal_quote
    ON proc.bp_sales_quote_line_internal (sales_quote_id);

INSERT INTO proc.bp_sales_quote_internal (sales_quote_id, total_cost, total_margin, margin_pct)
SELECT sales_quote_id, total_cost, total_margin, margin_pct
  FROM proc.bp_sales_quote
ON CONFLICT (sales_quote_id) DO NOTHING;

INSERT INTO proc.bp_sales_quote_line_internal
       (sales_quote_line_id, sales_quote_id, unit_cost, cost_tier_applied, line_margin, line_margin_pct)
SELECT sales_quote_line_id, sales_quote_id, unit_cost, cost_tier_applied, line_margin, line_margin_pct
  FROM proc.bp_sales_quote_line
ON CONFLICT (sales_quote_line_id) DO NOTHING;

COMMIT;
