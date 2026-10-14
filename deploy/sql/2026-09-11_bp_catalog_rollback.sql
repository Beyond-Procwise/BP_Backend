-- Rollback of 2026-09-11_bp_catalog.sql. Roll back bp_sell_side first: it references bp_catalog_item.
BEGIN;
DROP TABLE IF EXISTS proc.bp_catalog_item_match;
DROP TABLE IF EXISTS proc.bp_catalog_item_relation;
DROP TABLE IF EXISTS proc.bp_catalog_cost_tier;
DROP TABLE IF EXISTS proc.bp_catalog_item;
DROP TABLE IF EXISTS proc.bp_catalog_mapping;
DROP TABLE IF EXISTS proc.bp_catalog_source;
-- The key this migration added. Safe only once nothing references it.
ALTER TABLE proc.bp_supplier DROP CONSTRAINT IF EXISTS pk_bp_supplier;
COMMIT;
