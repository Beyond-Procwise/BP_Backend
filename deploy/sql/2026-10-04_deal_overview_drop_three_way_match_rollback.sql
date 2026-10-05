-- Restores `three_way_match` on proc.bp_deal_overview alongside the two new
-- columns (the state 2026-10-04_deal_overview_three_way.sql left), and rebuilds
-- bp_deal_kpis on it. Run this BEFORE rolling back any reader, so the column is
-- there when the old code asks for it.
BEGIN;
DROP VIEW IF EXISTS proc.bp_deal_overview CASCADE;
COMMIT;
-- Then, in order:
--   psql -f deploy/sql/2026-06-11_deal_views.sql
--   psql -f deploy/sql/2026-06-15_quote_anchor_views.sql
--   psql -f deploy/sql/2026-07-29_bp_deal_overview_value_reconciliation.sql
--   psql -f deploy/sql/2026-10-04_deal_overview_three_way.sql
