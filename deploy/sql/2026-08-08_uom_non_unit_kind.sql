-- Classify WHY each rejected value is not a unit, and settle the four
-- service-engagement bases.
--
-- DECISION (muthu, 2026-08-08): service, programme, retainer and audit are
-- rejected as units. They do NOT become an 'engagement' dimension.
--
-- The evidence, from proc.bp_product_master:
--
--   audit     -> 'HP | LaserJet Pro M404dn Mono Laser Printer'      £181.89
--   programme -> 'Logitech | MX Master 3S Wireless Mouse'            £86.29
--   programme -> 'Diversity, Equity & Inclusion Programme'         £8,900
--   retainer  -> 'HR Advisory & Employment Law Retainer (Quarterly)' £6,750
--   service   -> 'Payroll Managed Service (Quarterly)'              £4,200
--   service   -> 'Delivery & Installation Services'                 £4,500
--
-- Two different things hide in that list, and an 'engagement' dimension would
-- have been wrong about both:
--
--   * A printer is not sold "per audit" and a mouse is not sold "per
--     programme". Those two rows are extraction noise -- the word was lifted
--     from elsewhere on the page. Making the value a legitimate basis would
--     have quietly blessed a defect.
--   * For the genuine ones the real billing basis is stated in the
--     description: "(Quarterly)". The document says £6,750 per QUARTER, and
--     'quarter' is a unit we already map. Freezing "per retainer" as a basis
--     would enshrine a deliverable noun in place of the period that is
--     actually being billed.
--
-- And the reason that matters commercially: a basis exists to make two numbers
-- comparable. Two suppliers quoting "£50,000 per programme" have agreed on a
-- word, not on a quantity of anything. Treating that as a rate would let a
-- comparison return a confident wrong answer -- the exact failure measure_role
-- and basis_uom exist to prevent.
--
-- non_unit_kind records WHY a value was refused, because the four kinds need
-- different responses:
--
--   payment_term      -- extraction defect: belongs in payment_terms
--   scope_description -- extraction defect: belongs in the description
--   price_note        -- extraction defect: a price, or 'included', in the UoM column
--   deliverable_type  -- NOT a defect: the document really does say this. The
--                        line is a lump sum, and the true basis may be
--                        recoverable from the description (see "(Quarterly)")
--
-- The first three say "fix the pipeline". The fourth says "this is a real
-- lump-sum engagement" — and mixing them would have hidden a data-quality
-- problem inside a modelling one.
--
-- Idempotent and reversible.
BEGIN;

ALTER TABLE proc.bp_uom_canonical
    ADD COLUMN IF NOT EXISTS non_unit_kind text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_bp_uom_canonical_non_unit_kind'
          AND conrelid = 'proc.bp_uom_canonical'::regclass
    ) THEN
        ALTER TABLE proc.bp_uom_canonical
            ADD CONSTRAINT ck_bp_uom_canonical_non_unit_kind CHECK (
                non_unit_kind IS NULL OR non_unit_kind IN (
                    'payment_term', 'scope_description',
                    'price_note', 'deliverable_type'));
        -- An active unit is a unit; it cannot also be a reason for not being one.
        ALTER TABLE proc.bp_uom_canonical
            ADD CONSTRAINT ck_bp_uom_canonical_kind_only_when_not_a_unit CHECK (
                status <> 'active' OR non_unit_kind IS NULL);
    END IF;
END $$;

-- The decision: the four deliverable types are refused as units.
UPDATE proc.bp_uom_canonical
   SET status = 'rejected',
       dimension = NULL,
       is_billing_basis = false,
       non_unit_kind = 'deliverable_type',
       confirmed_by = 'muthu',
       confirmed_at = now()
 WHERE uom_code IN ('service', 'programme', 'retainer', 'audit')
   AND status <> 'rejected';

-- Classify the values already rejected.
UPDATE proc.bp_uom_canonical SET non_unit_kind = 'deliverable_type'
 WHERE non_unit_kind IS NULL AND status = 'rejected'
   AND uom_code IN ('service', 'programme', 'retainer', 'audit');

UPDATE proc.bp_uom_canonical SET non_unit_kind = 'payment_term'
 WHERE non_unit_kind IS NULL AND status = 'rejected'
   AND uom_code IN (
        '21 days from quote date', '30 days from quote date',
        '45 days from quote date', '30 days from invoice',
        '45 days from invoice', 'annual in advance',
        'annual in advance, 3-year pre-pay required');

UPDATE proc.bp_uom_canonical SET non_unit_kind = 'scope_description'
 WHERE non_unit_kind IS NULL AND status = 'rejected'
   AND uom_code IN (
        'transition 5 weeks', 'transition 6 weeks', 'transition 7 weeks',
        'transition 8 weeks', 'onboarding 7 weeks', 'onboarding 10 weeks');

UPDATE proc.bp_uom_canonical SET non_unit_kind = 'price_note'
 WHERE non_unit_kind IS NULL AND status = 'rejected'
   AND uom_code IN (
        'included',
        'implementation (one-off, fixed) — £58,000.00',
        'implementation (one-off, fixed) — £72,000.00');

COMMENT ON COLUMN proc.bp_uom_canonical.non_unit_kind IS
    'Why the value is not a unit. payment_term/scope_description/price_note '
    'are extraction defects; deliverable_type is not -- the document really '
    'does say it, and the line is a lump sum.';

COMMIT;
