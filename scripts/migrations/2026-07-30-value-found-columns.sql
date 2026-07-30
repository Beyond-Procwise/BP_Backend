-- Value Found (W1): distinguish HOW a discrepancy was resolved, and record the
-- Phase-3 "query sent" stamp. All nullable; historical resolved rows keep outcome
-- NULL and deliberately never count as "recovered".
ALTER TABLE proc.bp_extraction_discrepancy
    ADD COLUMN IF NOT EXISTS resolution_outcome TEXT,
    ADD COLUMN IF NOT EXISTS recovered_amount   NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS query_sent_at      TIMESTAMPTZ;

ALTER TABLE proc.bp_extraction_discrepancy
    DROP CONSTRAINT IF EXISTS bp_extraction_discrepancy_resolution_outcome_check;
ALTER TABLE proc.bp_extraction_discrepancy
    ADD CONSTRAINT bp_extraction_discrepancy_resolution_outcome_check
    CHECK (resolution_outcome IS NULL OR resolution_outcome IN ('recovered', 'accepted'));
