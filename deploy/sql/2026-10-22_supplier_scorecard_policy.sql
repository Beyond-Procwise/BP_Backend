-- Supplier scorecard: the on-time tolerances calc_supplier_scorecard uses are a governed rule, not a default.
-- FOR REVIEW. NOT APPLIED to any database. Idempotent: writes the row only if there is none, so it never
-- overwrites an administrator's choice. Until this row exists the tool reports on-time delivery as
-- "not assessed" and names this policy as the reason; spend, price variance and findings work without it.
--
-- Rules (read by name through services.governed_limits.rules('supplier_scorecard')):
--   otif_late_tolerance_days   days after the PO's expected delivery date a receipt still counts on time
--   otif_early_tolerance_days  days before it a receipt still counts on time; null = early is not penalised
--   otif_in_full_tolerance     share short of the ordered quantity still counted in full (unused until
--                              received quantities are recorded; kept so the formula's inputs are complete)
-- Deliberately NO applies_to: configuration read by name, not gate-visible authority.
-- After applying, add the same rules to GOVERNED_LIMIT_SEED in tests/conftest.py (the drift test compares
-- the seed to the live rows and fails while they differ).
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version, created_date, created_by
)
SELECT 'SupplierScorecardLimitPolicy', 'limit',
       'Tolerances for the supplier scorecard''s on-time delivery measure: how many days late (or early) a goods receipt may land against the PO''s expected delivery date and still count as on time.',
       jsonb_build_object('policy_identifier', 'supplier_scorecard',
                          'rules', jsonb_build_object('otif_late_tolerance_days', 0,
                                                      'otif_early_tolerance_days', NULL,
                                                      'otif_in_full_tolerance', 0.0)),
       '', 1, 1, now(), 'migration:2026-10-22_supplier_scorecard_policy'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'SupplierScorecardLimitPolicy' AND policy_status = 1
);

COMMIT;
