-- Reverses 2026-09-09_opportunity_critique.sql. Gaps first: they reference
-- critiques.
BEGIN;
DROP TABLE IF EXISTS proc.bp_opportunity_gap;
DROP TABLE IF EXISTS proc.bp_opportunity_critique;
COMMIT;
