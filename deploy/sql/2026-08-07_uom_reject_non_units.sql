-- Confirm the 14 values that are not units.
--
-- These are payment terms, scope descriptions and prices that landed in the
-- unit_of_measure column. Phase 1b's F5 established the set and the plan
-- required that all 14 be refused rather than coerced; this records that
-- decision as data so the reconciler stops re-proposing them.
--
-- Authorised explicitly by muthu, 2026-08-07.
--
-- Listed one by one ON PURPOSE. The tempting version is
--
--     UPDATE ... SET status='rejected'
--      WHERE status='proposed' AND uom_code NOT IN ('service', ...)
--
-- which is wrong in a way that would not show up today: it rejects whatever
-- happens to be proposed at the moment it runs. Re-run next month, after the
-- reconciler has queued a genuine new unit, and it silently rejects that too.
-- An explicit list can only ever do what it says.
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
