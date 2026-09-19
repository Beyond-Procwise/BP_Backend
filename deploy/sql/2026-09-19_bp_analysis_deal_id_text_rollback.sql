-- Narrowing back refuses (and changes nothing) once any analysis has been filed
-- against a deal id longer than 25 characters -- which is the case this fixed.
BEGIN;

DO $$
BEGIN
  IF to_regclass('proc.bp_analysis_deal') IS NOT NULL THEN
    ALTER TABLE proc.bp_analysis_deal ALTER COLUMN deal_id TYPE VARCHAR(25);
  END IF;
END $$;

COMMIT;
