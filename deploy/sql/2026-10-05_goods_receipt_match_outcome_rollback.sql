-- Reverses deploy/sql/2026-10-05_goods_receipt_match_outcome.sql.
-- Re-apply 2026-10-04_deal_overview_drop_three_way_match.sql FIRST in its
-- pre-2026-10-05 form, or the view will reference a column that is gone.
BEGIN;
ALTER TABLE proc.bp_goods_receipt_trgt DROP COLUMN IF EXISTS lines_unverifiable;
ALTER TABLE proc.bp_goods_receipt_trgt DROP COLUMN IF EXISTS lines_assessed;
ALTER TABLE proc.bp_goods_receipt_stg  DROP COLUMN IF EXISTS lines_unverifiable;
ALTER TABLE proc.bp_goods_receipt_stg  DROP COLUMN IF EXISTS lines_assessed;
COMMIT;
