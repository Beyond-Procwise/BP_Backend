-- The Quote builder's quote (UI BACKEND_GAPS.md, "Quote builder").
--
-- The sell-side quote was built for a reseller: it is addressed to an account, every
-- line is a catalog item, and a line is never edited. The Quote builder drafts a
-- different-shaped offer on the same record: it belongs to a deal (or to none yet),
-- its lines are written by the seller with the seller's own cost, and a draft is saved
-- many times before anyone reviews it. These columns let one record hold both, so
-- approval, numbering, the customer page, sending and the audit trail stay shared.
--
--   * account_id becomes optional: a builder quote names its customer in words
--     (customer_name) until an account exists. A quote with no account still cannot be
--     SENT -- sending needs a contact on an account -- so nothing reaches a customer
--     through this gap.
--   * deal_id is set once and one deal has one live quote. The unique index is the
--     control; the service check in front of it only gives the better message.
--   * entered_discount_pct is the discount the seller typed, in percent (4 = 4%). It
--     is NOT discount_pct, which is derived from the catalog list price and is a
--     fraction. The two never share a column.
--   * cost_source says whose figure unit_cost is: 'catalog' (read from the price
--     file) or 'entered' (typed by the seller). Rows written before today are catalog.
--   * costed_lines / uncosted_lines: a builder quote's margin is taken over the lines
--     that have a cost, and these say how many were left out. NULL on a catalog quote,
--     whose margin is all-or-nothing.
--
-- Additive and idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code that
-- reads it.

BEGIN;

ALTER TABLE proc.bp_sales_quote
    ALTER COLUMN account_id DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS deal_id          text,
    ADD COLUMN IF NOT EXISTS version          integer NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS customer_name    text,
    ADD COLUMN IF NOT EXISTS customer_contact text,
    ADD COLUMN IF NOT EXISTS category         text,
    ADD COLUMN IF NOT EXISTS prepared_by      text,
    ADD COLUMN IF NOT EXISTS scope_html       text,
    ADD COLUMN IF NOT EXISTS assumptions_html text,
    ADD COLUMN IF NOT EXISTS terms            jsonb,
    ADD COLUMN IF NOT EXISTS conditions       jsonb NOT NULL DEFAULT '[]'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_sales_quote_deal_live
    ON proc.bp_sales_quote (deal_id)
    WHERE deal_id IS NOT NULL AND status <> 'superseded';

ALTER TABLE proc.bp_sales_quote_line
    ADD COLUMN IF NOT EXISTS line_tag             text,
    ADD COLUMN IF NOT EXISTS entered_discount_pct numeric(7,4);

ALTER TABLE proc.bp_sales_quote_line_internal
    ADD COLUMN IF NOT EXISTS cost_source text,
    ADD COLUMN IF NOT EXISTS markup_pct  numeric(11,4);

UPDATE proc.bp_sales_quote_line_internal
   SET cost_source = 'catalog'
 WHERE cost_source IS NULL AND unit_cost IS NOT NULL;

ALTER TABLE proc.bp_sales_quote_internal
    ADD COLUMN IF NOT EXISTS costed_lines   integer,
    ADD COLUMN IF NOT EXISTS uncosted_lines integer;

COMMIT;
