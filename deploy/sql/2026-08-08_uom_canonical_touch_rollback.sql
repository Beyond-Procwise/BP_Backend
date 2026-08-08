-- Rollback for 2026-08-08_uom_canonical_touch.sql
--
-- ORDERING WARNING. Drop the trigger before the function it calls. Dropping
-- the function first fails while the trigger still depends on it, and with
-- CASCADE it would take the trigger with it silently — leaving a rollback that
-- appears to have worked either way.
--
-- Not destructive to data. Existing recorded_at values are left as they are;
-- only the automatic maintenance stops.
BEGIN;

DROP TRIGGER IF EXISTS bp_uom_canonical_touch ON proc.bp_uom_canonical;
DROP FUNCTION IF EXISTS proc.bp_uom_canonical_touch();

COMMIT;
