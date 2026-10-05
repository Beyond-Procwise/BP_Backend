-- Reverses deploy/sql/2026-10-04_goods_receipt_tables.sql.
-- Children first: the line_items_raw table references bp_goods_receipt_raw.
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_trgt;
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_stg;
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_raw;
DROP TABLE IF EXISTS proc.bp_goods_receipt_trgt;
DROP TABLE IF EXISTS proc.bp_goods_receipt_stg;
DROP TABLE IF EXISTS proc.bp_goods_receipt_raw;
