-- Rollback for 2026-10-13_opportunity_subject_review.sql.
--
-- Run only with the code that reads these columns already rolled back. A row
-- re-opened from 'rejected' stays where a person put it; removing the transition
-- only stops further re-opening.
BEGIN;

DELETE FROM proc.bp_lifecycle_transition
 WHERE object_type = 'opportunity' AND from_state = 'rejected' AND to_state = 'identified';

DROP INDEX IF EXISTS proc.ix_bp_opportunity_review_requested;
DROP INDEX IF EXISTS proc.ux_bp_opportunity_subject;

ALTER TABLE proc.bp_opportunity
    DROP COLUMN IF EXISTS review_detail,
    DROP COLUMN IF EXISTS review_requested_at,
    DROP COLUMN IF EXISTS subject_key;

COMMIT;
