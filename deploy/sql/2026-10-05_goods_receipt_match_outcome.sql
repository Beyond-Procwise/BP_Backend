-- 2026-10-05  A receipt records what the match could and could not assess.
--
-- Found in the whole-branch review: check() computed `unverifiable` and
-- `assessed` correctly and then THREW THEM AWAY. check_against_receipts
-- iterated the findings only, so a refusal reached no reader, and the deal
-- overview's verdict ("a receipt with lines exists AND no open gap finding")
-- turned every refusal into `three_way_matched = true`. A deal whose note
-- counts in `each` against an order in `box` -- the design's own "single most
-- likely practical failure" -- read as "Goods billed were received: 100%".
--
-- A refusal nobody can see is a pass. So the counts are persisted and the
-- verdict is gated on them:
--
--   lines_assessed      PO lines where ordered, received and billed were all
--                       genuinely comparable. This is the DENOMINATOR section
--                       13.3 of the design asks for, and the view now requires
--                       it to be greater than zero before giving a verdict.
--   lines_unverifiable  PO lines that could not be checked, with the reason on
--                       each line's own discrepancy row.
--
-- Neither is a price. Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_goods_receipt_stg
  ADD COLUMN IF NOT EXISTS lines_assessed integer;
ALTER TABLE proc.bp_goods_receipt_stg
  ADD COLUMN IF NOT EXISTS lines_unverifiable integer;
ALTER TABLE proc.bp_goods_receipt_trgt
  ADD COLUMN IF NOT EXISTS lines_assessed integer;
ALTER TABLE proc.bp_goods_receipt_trgt
  ADD COLUMN IF NOT EXISTS lines_unverifiable integer;

CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_trgt_lines_assessed
  ON proc.bp_goods_receipt_trgt (lines_assessed);

COMMIT;
