-- Rollback for 2026-08-08_uom_non_unit_kind.sql
--
-- ORDERING WARNING. Drop both CHECK constraints before the column they
-- reference. Dropping the column first removes its constraints implicitly,
-- which succeeds once and then leaves the DROP CONSTRAINT statements to fail
-- on a re-run -- so the rollback stops being idempotent exactly when it is
-- most likely to be re-run.
--
-- Returns the four deliverable types to 'proposed', so they re-enter the
-- review queue rather than silently disappearing from it.
BEGIN;

UPDATE proc.bp_uom_canonical
   SET status = 'proposed',
       confirmed_by = NULL,
       confirmed_at = NULL
 WHERE status = 'rejected'
   AND non_unit_kind = 'deliverable_type'
   AND uom_code IN ('service', 'programme', 'retainer', 'audit');

ALTER TABLE proc.bp_uom_canonical
    DROP CONSTRAINT IF EXISTS ck_bp_uom_canonical_kind_only_when_not_a_unit;
ALTER TABLE proc.bp_uom_canonical
    DROP CONSTRAINT IF EXISTS ck_bp_uom_canonical_non_unit_kind;

ALTER TABLE proc.bp_uom_canonical
    DROP COLUMN IF EXISTS non_unit_kind;

COMMIT;
