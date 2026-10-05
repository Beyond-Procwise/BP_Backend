-- 2026-10-04  How far delivery may differ from order and invoice before it is a finding.
--
-- The three-way match reads these through governed_limits.limit(), which RAISES
-- when a rule is absent. That is deliberate (project_governed_limits): there is
-- no code-side default, because a default is a number nobody agreed to that
-- silently decides whether a supplier is over-billing. With no row here the
-- match refuses to run and says so; with a wrong row here somebody can see the
-- number and change it.
--
-- The opening values are ZERO tolerance on both:
--   billed_over_received_qty = 0   one unit billed beyond what arrived is a
--                                  finding. Quantities are integers in practice
--                                  and there is no rounding to absorb.
--   over_delivery_pct        = 0   a supplier delivering more than was ordered
--                                  is worth a look even at one unit; it is a
--                                  warning, not an over-billing.
-- They are a starting position, not a measurement: there are zero goods receipts
-- in the corpus (design section 13), so no tolerance can be sized from evidence
-- yet. Zero is the honest opening -- it over-reports rather than under-reports,
-- and it is a row somebody can widen once there is data.
--
-- uom_conversion_required = true says the match must REFUSE a line whose
-- receipt unit differs from its order unit rather than guess a conversion.
-- Setting it false would not enable conversion; nothing converts. It exists so
-- that whoever one day adds conversion has a governed switch to add it behind.
--
-- Idempotent. Safe to re-run.

BEGIN;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details, policy_linked_agents,
     policy_status, version, created_by, last_modified_by)
SELECT
    'ReceiptTolerancePolicy',
    -- policy_type = 'limit', NOT 'extraction' as the plan wrote it. Every other
    -- governed-limit policy is 'limit', and more importantly
    -- tests/governance/test_governed_limits.py compares the LIVE 'limit' rows
    -- against the copy in tests/conftest.py and fails when they drift. A row
    -- outside that type escapes the only check that keeps the copy honest.
    'limit',
    'How far delivery may differ from order and invoice before it is a finding.',
    '{"policy_identifier": "receipt_tolerances",
      "applies_to": ["three_way_match"],
      "rules": {"over_delivery_pct": 0.00,
                "billed_over_received_qty": 0,
                "uom_conversion_required": true}}'::jsonb,
    'data_extraction',
    1,
    1,
    'receipt_tolerances_policy_migration_2026_10_04',
    'receipt_tolerances_policy_migration_2026_10_04'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
     WHERE policy_details->>'policy_identifier' = 'receipt_tolerances'
);

COMMIT;
