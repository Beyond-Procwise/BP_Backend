-- Rollback for 2026-08-07_uom_reject_non_units.sql
--
-- Returns the 14 non-unit values to 'proposed' and clears the confirmation,
-- so they re-enter the review queue rather than vanishing.
--
-- Scoped to status='rejected' AND this exact list, so it cannot resurrect a
-- rejection somebody made for a different reason later.
BEGIN;

UPDATE proc.bp_uom_canonical
   SET status = 'proposed',
       confirmed_by = NULL,
       confirmed_at = NULL,
       recorded_at = now()
 WHERE status = 'rejected'
   AND uom_code IN (
        '21 days from quote date',
        '30 days from invoice',
        '30 days from quote date',
        '45 days from invoice',
        '45 days from quote date',
        'annual in advance',
        'implementation (one-off, fixed) — £58,000.00',
        'implementation (one-off, fixed) — £72,000.00',
        'included',
        'onboarding 10 weeks',
        'onboarding 7 weeks',
        'transition 5 weeks',
        'transition 6 weeks',
        'transition 7 weeks'
   );

COMMIT;
