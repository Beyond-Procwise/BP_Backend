-- Rollback for 2026-10-05_contract_expiry_bucket_rule.sql
BEGIN;
DELETE FROM proc.bp_rule WHERE detector_slug = 'contract_expiry_bucket_check';
DROP TABLE IF EXISTS proc.bp_contract_expiry_alert;
COMMIT;
