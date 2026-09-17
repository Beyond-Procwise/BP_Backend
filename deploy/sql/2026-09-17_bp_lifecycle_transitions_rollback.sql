BEGIN;
-- deploy/sql/2026-09-17_bp_lifecycle_transitions_rollback.sql
--
-- Reverses 2026-09-17_bp_lifecycle_transitions.sql.
--
-- Running this lets any writer move a finding or an opportunity to any state again: a
-- realised saving can be unwound to identified, and a second person's click silently
-- replaces the first person's resolution. Rows are untouched by the rollback itself.
--
-- The Python writers keep filtering on proc.bp_lifecycle_transition, so the table is
-- dropped LAST and only after the triggers; if you drop it, deploy the code that
-- predates it too, or those writers will fail on a missing relation.

DROP TRIGGER IF EXISTS tr_bp_extraction_discrepancy_lifecycle ON proc.bp_extraction_discrepancy;
DROP TRIGGER IF EXISTS tr_bp_opportunity_lifecycle ON proc.bp_opportunity;
DROP FUNCTION IF EXISTS proc.bp_lifecycle_guard();
DROP TABLE IF EXISTS proc.bp_lifecycle_transition;

COMMIT;
