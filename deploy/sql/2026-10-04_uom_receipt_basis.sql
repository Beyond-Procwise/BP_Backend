-- Which units can be RECEIVED at all.
--
-- `dimension` describes the physical measure; `receipt_basis` describes whether
-- a human can take delivery of what the unit measures. They disagree on
-- licence, seat and module -- all dimension 'count', none of them deliverable --
-- which is why this is a column and not a view over dimension. A three-way
-- match reading `dimension` would expect a goods receipt for a software licence,
-- find none, and report the line unreceived: a manufactured failure, and the
-- design's §6 exists to prevent exactly that.
--
-- Three values, and they mean:
--   goods_receipt  a delivery note can prove this line          (this feature)
--   service_entry  only a service entry sheet could             (out of scope)
--   none           not a unit at all; nothing can prove it
--
-- The DEFAULT is 'none', deliberately. A unit this migration has never seen --
-- 'pallet' learned from a future document -- then reads as UNVERIFIABLE rather
-- than as received or as missing. Under-claiming is the safe direction: it
-- shows up in the not-assessed denominator where somebody can see it, instead
-- of becoming a pass or a failure nobody asked for. The default is set AFTER
-- the backfill so the IS NULL predicates below still find the existing rows.
--
-- Additive, idempotent, reversible.
BEGIN;

ALTER TABLE proc.bp_uom_canonical ADD COLUMN IF NOT EXISTS receipt_basis text;

UPDATE proc.bp_uom_canonical SET receipt_basis = 'goods_receipt'
 WHERE dimension IN ('count','mass','length','volume') AND receipt_basis IS NULL;

UPDATE proc.bp_uom_canonical SET receipt_basis = 'service_entry'
 WHERE dimension = 'time' AND receipt_basis IS NULL;

-- The intangible counts, corrected by hand. Counted, but nobody takes delivery
-- of one.
UPDATE proc.bp_uom_canonical SET receipt_basis = 'service_entry'
 WHERE uom_code IN ('licence','license','seat','module','subscription','user');

-- The 18 rows whose dimension is NULL are extraction noise that reached a
-- reference table ('30 days from invoice', 'implementation (one-off, fixed) --
-- GBP 58,000.00'). They are not units, and they are NOT classified as a
-- receipt basis of their own: 'none' says nothing can prove them.
--
-- The plan asked for status = 'retired' here. It is not applied, for two
-- reasons: these rows are ALREADY status = 'rejected' and tagged with a
-- non_unit_kind by 2026-08-08_uom_non_unit_kind.sql, which is the house
-- vocabulary for "this is not a unit"; and 'retired' is not in
-- ck_bp_uom_canonical_status, so the statement simply fails.
UPDATE proc.bp_uom_canonical SET receipt_basis = 'none'
 WHERE dimension IS NULL;

ALTER TABLE proc.bp_uom_canonical ALTER COLUMN receipt_basis SET DEFAULT 'none';

-- NOT VALID: the backfill above has already classified every existing row, and
-- an unvalidated constraint still polices every INSERT and UPDATE from here on,
-- which is the point. A fourth spelling would be read by the match's branch as
-- "not goods_receipt" and silently excluded.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'ck_bp_uom_canonical_receipt_basis'
           AND conrelid = 'proc.bp_uom_canonical'::regclass
    ) THEN
        ALTER TABLE proc.bp_uom_canonical
          ADD CONSTRAINT ck_bp_uom_canonical_receipt_basis
          CHECK (receipt_basis IN ('goods_receipt','service_entry','none')) NOT VALID;
    END IF;
END $$;

COMMIT;
