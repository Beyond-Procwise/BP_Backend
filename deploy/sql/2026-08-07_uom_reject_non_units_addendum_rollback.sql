-- Rollback for 2026-08-07_uom_reject_non_units_addendum.sql
--
-- Returns the two values to 'proposed' and clears the confirmation, so they
-- re-enter the review queue rather than vanishing.
--
-- Scoped to status='rejected' AND this exact pair, so it cannot resurrect a
-- rejection made later for a different reason.
BEGIN;

UPDATE proc.bp_uom_canonical
   SET status = 'proposed',
       confirmed_by = NULL,
       confirmed_at = NULL,
       recorded_at = now()
 WHERE status = 'rejected'
   AND uom_code IN (
        'annual in advance, 3-year pre-pay required',
        'transition 8 weeks'
   );

COMMIT;
