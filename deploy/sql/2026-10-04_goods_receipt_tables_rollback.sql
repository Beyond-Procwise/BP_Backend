-- Reverses deploy/sql/2026-10-04_goods_receipt_tables.sql.
--
-- ORDER MATTERS AND IS NOT OPTIONAL. proc.bp_deal_overview reads
-- proc.bp_goods_receipt_trgt (its `receipted` CTE, added by
-- 2026-10-04_deal_overview_three_way.sql), and proc.bp_deal_kpis reads the
-- overview -- so these DROPs fail outright until the view has gone back to a
-- definition that does not mention a goods receipt:
--
--   1. psql -f deploy/sql/2026-10-04_deal_overview_three_way_rollback.sql
--      (which itself CASCADEs bp_deal_kpis -- read its header)
--   2. psql -f deploy/sql/2026-06-11_deal_views.sql
--   3. psql -f deploy/sql/2026-06-15_quote_anchor_views.sql
--   4. then this file
--
-- Within this file, children first: bp_goods_receipt_line_items_raw carries a
-- foreign key onto bp_goods_receipt_raw.
--
-- Measured 2026-10-05: run out of order on bp_sqldb this fails with
-- "cannot drop table proc.bp_goods_receipt_trgt because other objects depend
-- on it", which is the right failure -- it refuses rather than CASCADEing the
-- deal overview and the KPI view away behind your back.
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_trgt;
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_stg;
DROP TABLE IF EXISTS proc.bp_goods_receipt_line_items_raw;
DROP TABLE IF EXISTS proc.bp_goods_receipt_trgt;
DROP TABLE IF EXISTS proc.bp_goods_receipt_stg;
DROP TABLE IF EXISTS proc.bp_goods_receipt_raw;
