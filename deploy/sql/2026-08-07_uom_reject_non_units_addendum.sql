-- Two further values that are not units, found by the reconciler.
--
-- Phase 1b's F5 enumerated 28 distinct unit_of_measure values and classified
-- 14 of them as junk. It missed these two, which exist in bp_sqldb's corpus:
--
--   'annual in advance, 3-year pre-pay required'  -- a payment term
--   'transition 8 weeks'                          -- a scope description;
--                                                    F5 catalogued the 5, 6 and
--                                                    7 week variants but not 8
--
-- Worth recording why that matters beyond these two rows: a careful manual
-- enumeration missed them, and scripts/reconcile_uom_vocabulary.py found them
-- on its first run. Any list of "the values in the corpus" is a snapshot; the
-- counter is what keeps it true.
--
-- Authorised explicitly by muthu, 2026-08-07.
--
-- Applied to both databases even though these rows exist only on bp_sqldb, so
-- the two stay identical in behaviour: the UPDATE simply matches nothing where
-- the values are absent, rather than the schemas quietly diverging.
--
-- Idempotent and reversible.
BEGIN;

UPDATE proc.bp_uom_canonical
   SET status = 'rejected',
       dimension = NULL,
       is_billing_basis = false,
       confirmed_by = 'muthu',
       confirmed_at = now(),
       recorded_at = now()
 WHERE status = 'proposed'
   AND uom_code IN (
        'annual in advance, 3-year pre-pay required',
        'transition 8 weeks'
   );

COMMIT;
