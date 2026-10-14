-- Reverses deploy/sql/2026-10-04_receipt_tolerances_policy.sql.
-- With this row gone the three-way match RAISES rather than assuming a
-- tolerance. That is the intended behaviour, not a regression.
DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'receipt_tolerances';
