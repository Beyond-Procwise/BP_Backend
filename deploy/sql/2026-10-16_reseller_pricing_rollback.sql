-- Rollback of 2026-10-16_reseller_pricing.sql. Drops the agreements and the provenance columns.
BEGIN;
ALTER TABLE proc.bp_sales_quote_line_internal
    DROP COLUMN IF EXISTS price_source, DROP COLUMN IF EXISTS account_price_id,
    DROP COLUMN IF EXISTS cost_source,  DROP COLUMN IF EXISTS special_bid_id;
DROP TABLE IF EXISTS proc.bp_catalog_special_bid;
DROP TABLE IF EXISTS proc.bp_account_price;
COMMIT;
