-- An analysis can be filed against any deal the gateway mints.
--
-- bp_analysis_deal.deal_id was VARCHAR(25), but the gateway builds a deal id
-- from the name the user types plus the date and a counter, and proc.bp_deal
-- holds it as TEXT. Any name over ~10 characters produced a longer id, so
-- analysis_store.freeze() failed in _link_deals ("value too long for type
-- character varying(25)"), the analysis stayed 'running', and the report sat
-- on "Reading your documents…" for good. ORB_2026-09-18 (24 chars) completed;
-- SaaStest_2026-09-18 (29) and ORB_Analysis_2026-09-19 (34) did not.
--
-- TEXT matches proc.bp_deal.deal_id. Widening needs no rewrite and keeps the
-- primary key and the (deal_id, version) unique constraint as they are.
-- A database without the analysis tables is left alone.
BEGIN;

DO $$
BEGIN
  IF to_regclass('proc.bp_analysis_deal') IS NOT NULL THEN
    ALTER TABLE proc.bp_analysis_deal ALTER COLUMN deal_id TYPE TEXT;
  END IF;
END $$;

COMMIT;
