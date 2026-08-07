-- Rollback for 2026-08-07_commercial_fact.sql
--
-- ORDERING WARNING. Drop in foreign-key-safe order: the two constraint
-- triggers first, then the tables that reference bp_commercial_fact
-- (bp_finding_fact, bp_fact_provenance), then bp_constraint, and only then
-- bp_commercial_fact itself. Dropping bp_commercial_fact first would either
-- fail on the dependent foreign keys or, with CASCADE, silently take the
-- dependants with it -- which looks like a clean rollback right up until the
-- re-apply produces a different schema from the one you dropped.
--
-- Destructive: this discards every assembled fact and its provenance.
BEGIN;

DROP TRIGGER IF EXISTS bp_fact_provenance_keeps_fact_covered
    ON proc.bp_fact_provenance;
DROP TRIGGER IF EXISTS bp_commercial_fact_provenance_required
    ON proc.bp_commercial_fact;

DROP TABLE IF EXISTS proc.bp_finding_fact;
DROP TABLE IF EXISTS proc.bp_fact_provenance;
DROP TABLE IF EXISTS proc.bp_constraint;
DROP TABLE IF EXISTS proc.bp_commercial_fact;

DROP FUNCTION IF EXISTS proc.bp_fact_provenance_keep_fact_covered();
DROP FUNCTION IF EXISTS proc.bp_commercial_fact_require_provenance();

COMMIT;
