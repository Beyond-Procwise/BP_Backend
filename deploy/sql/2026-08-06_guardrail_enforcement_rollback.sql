BEGIN;

DELETE FROM proc.bp_policy
WHERE policy_details->>'policy_identifier' IN (
    'role_definition',
    'role_assignment',
    'email_dispatch_approval',
    'email_recipient_allowlist',
    'email_sensitivity',
    'email_volume'
);

DROP INDEX IF EXISTS proc.ix_bp_role_assignment_subject;
DROP TABLE IF EXISTS proc.bp_role_assignment;

ALTER TABLE proc.bp_supplier DROP COLUMN IF EXISTS clearance_level;

COMMIT;
