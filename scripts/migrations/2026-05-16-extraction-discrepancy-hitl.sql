-- scripts/migrations/2026-05-16-extraction-discrepancy-hitl.sql
-- Additive: extend proc.bp_extraction_discrepancy with HITL workflow columns
-- and add the re-promotion NOTIFY trigger.
--
-- Existing 1,543 discrepancy rows are left untouched. The new
-- blocks_promotion column defaults to FALSE so existing rows do not
-- participate in the new NOTIFY path. New code path writes blocks_promotion
-- based on severity (critical → TRUE).
--
-- Trigger condition: NOTIFY fires only when (a) status transitions to
-- 'resolved', (b) the row being resolved had blocks_promotion=TRUE, and
-- (c) no other blocking-open discrepancies remain for the same raw_id.
--
-- Run against: bp_sqldb.

BEGIN;

-- 1. Extend the discrepancy table for HITL workflow
ALTER TABLE proc.bp_extraction_discrepancy
    ADD COLUMN IF NOT EXISTS resolved_value     TEXT,
    ADD COLUMN IF NOT EXISTS resolution_action  TEXT,
    ADD COLUMN IF NOT EXISTS resolved_by        TEXT,
    ADD COLUMN IF NOT EXISTS blocks_promotion   BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS evidence_page      INT,
    ADD COLUMN IF NOT EXISTS evidence_bbox      REAL[],
    ADD COLUMN IF NOT EXISTS evidence_text      TEXT;

-- 2. resolution_action enum (drop-and-add for idempotence)
ALTER TABLE proc.bp_extraction_discrepancy
    DROP CONSTRAINT IF EXISTS bp_extraction_discrepancy_resolution_action_check;
ALTER TABLE proc.bp_extraction_discrepancy
    ADD CONSTRAINT bp_extraction_discrepancy_resolution_action_check
    CHECK (resolution_action IS NULL
        OR resolution_action IN ('apply_value', 'keep_null', 'dismiss'));

-- 3. Extend status enum to include 'superseded'
ALTER TABLE proc.bp_extraction_discrepancy
    DROP CONSTRAINT IF EXISTS bp_extraction_discrepancy_status_check;
ALTER TABLE proc.bp_extraction_discrepancy
    ADD CONSTRAINT bp_extraction_discrepancy_status_check
    CHECK (status IN ('open', 'resolved', 'ignored', 'superseded'));

-- 4. Re-promotion NOTIFY trigger
CREATE OR REPLACE FUNCTION proc.fn_extraction_discrepancy_resolved()
RETURNS TRIGGER AS $$
DECLARE
    open_blocking_count INT;
BEGIN
    IF NEW.status = 'resolved'
       AND (OLD.status IS DISTINCT FROM NEW.status)
       AND NEW.blocks_promotion = TRUE THEN
        SELECT COUNT(*) INTO open_blocking_count
          FROM proc.bp_extraction_discrepancy
         WHERE raw_id = NEW.raw_id
           AND blocks_promotion = TRUE
           AND status = 'open';

        IF open_blocking_count = 0 THEN
            PERFORM pg_notify(
                'extraction_raw_ready_for_promotion',
                json_build_object('doc_type', NEW.doc_type, 'raw_id', NEW.raw_id)::text
            );
        END IF;
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_extraction_discrepancy_resolved
    ON proc.bp_extraction_discrepancy;
CREATE TRIGGER trg_extraction_discrepancy_resolved
AFTER UPDATE ON proc.bp_extraction_discrepancy
FOR EACH ROW EXECUTE FUNCTION proc.fn_extraction_discrepancy_resolved();

COMMIT;
