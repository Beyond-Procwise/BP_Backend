-- Reseller pricing: what a customer has agreed to pay, and what a distributor has agreed
-- to charge us for one named deal. Stage 3 items 1 and 2 of the reseller review.
--
-- Both are keyed by (distributor_id, distributor_sku), NOT catalog_item_id: a catalog
-- reprice closes a version and opens a new row with a new id, which would orphan every
-- agreement that pointed at the old one.
--
--   bp_account_price       a customer's agreed price: a fixed unit price or a discount off
--                          list, optionally from a quantity, for a dated window.
--   bp_catalog_special_bid a distributor's deal cost for one customer: bid reference,
--                          cost, quantity band, and a MANDATORY expiry.
--
-- A quote line records where its price and cost came from (price_source / cost_source and
-- the row ids), in the internal line table, so it can be reconstructed later.
--
-- Idempotent. Apply to BOTH bp_testdb and bp_sqldb. Rollback: _rollback.sql beside this file.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_account_price (
    account_price_id  bigserial PRIMARY KEY,
    tenant_id         text,
    account_id        text NOT NULL REFERENCES proc.bp_account (account_id) ON DELETE CASCADE,
    distributor_id    text NOT NULL,
    distributor_sku   text NOT NULL,
    price_kind        text NOT NULL,                 -- fixed | discount_pct
    unit_price        numeric(18,4),
    discount_pct      numeric(7,4),
    currency          char(3) NOT NULL,
    min_quantity      numeric NOT NULL DEFAULT 0,
    valid_from        date NOT NULL,
    valid_to          date,
    reference         text,                          -- the contract or price-list it comes from
    status            text NOT NULL DEFAULT 'active',  -- active | withdrawn
    created_by        text,
    created_date      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_account_price_kind CHECK (
        (price_kind = 'fixed' AND unit_price IS NOT NULL AND unit_price >= 0
            AND discount_pct IS NULL)
     OR (price_kind = 'discount_pct' AND discount_pct IS NOT NULL
            AND discount_pct >= 0 AND discount_pct < 100 AND unit_price IS NULL)),
    CONSTRAINT ck_bp_account_price_status CHECK (status IN ('active', 'withdrawn')),
    CONSTRAINT ck_bp_account_price_window CHECK (valid_to IS NULL OR valid_to >= valid_from),
    CONSTRAINT ck_bp_account_price_qty CHECK (min_quantity >= 0)
);
CREATE INDEX IF NOT EXISTS ix_bp_account_price_account_sku
    ON proc.bp_account_price (account_id, distributor_id, distributor_sku);

CREATE TABLE IF NOT EXISTS proc.bp_catalog_special_bid (
    special_bid_id    bigserial PRIMARY KEY,
    tenant_id         text,
    account_id        text NOT NULL REFERENCES proc.bp_account (account_id) ON DELETE CASCADE,
    distributor_id    text NOT NULL,
    distributor_sku   text NOT NULL,
    bid_reference     text NOT NULL,                 -- the distributor's registration number
    cost_price        numeric(18,4) NOT NULL,
    currency          char(3) NOT NULL,
    min_quantity      numeric NOT NULL DEFAULT 0,
    max_quantity      numeric,                       -- the quantity the distributor approved
    valid_from        date NOT NULL,
    valid_to          date NOT NULL,                 -- a bid always expires
    status            text NOT NULL DEFAULT 'active',  -- active | withdrawn
    created_by        text,
    created_date      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_special_bid_cost CHECK (cost_price >= 0),
    CONSTRAINT ck_bp_special_bid_status CHECK (status IN ('active', 'withdrawn')),
    CONSTRAINT ck_bp_special_bid_window CHECK (valid_to >= valid_from),
    CONSTRAINT ck_bp_special_bid_qty CHECK (
        min_quantity >= 0 AND (max_quantity IS NULL OR max_quantity >= min_quantity))
);
CREATE INDEX IF NOT EXISTS ix_bp_catalog_special_bid_account_sku
    ON proc.bp_catalog_special_bid (account_id, distributor_id, distributor_sku);

ALTER TABLE proc.bp_sales_quote_line_internal
    ADD COLUMN IF NOT EXISTS price_source      text,     -- entered | agreed_price
    ADD COLUMN IF NOT EXISTS account_price_id  bigint,
    ADD COLUMN IF NOT EXISTS cost_source       text,     -- catalog | special_bid
    ADD COLUMN IF NOT EXISTS special_bid_id    bigint;

COMMIT;
