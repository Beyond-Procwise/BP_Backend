-- 2026-07-29 Record WHICH earlier upload a content-duplicate matched.
--
-- Uploading documents that are already in the system marks them
-- doc_action='duplicate' and skips re-extraction, which is right: the same invoice
-- on two deals would double-count in every spend, savings and compliance figure.
-- But the analysis then rendered "0 documents analysed · on track · no open
-- findings" with no indication of where the documents actually live — the user is
-- told nothing was added, not which deal already holds them.
--
-- The prior process_monitor row is already identified at detection time; this
-- persists it so the deal it belongs to can be named on screen.
--
-- Additive + idempotent.
BEGIN;

ALTER TABLE proc.process_monitor
    ADD COLUMN IF NOT EXISTS duplicate_of_id BIGINT;

COMMENT ON COLUMN proc.process_monitor.duplicate_of_id IS
    'When doc_action = ''duplicate'', the process_monitor.id of the earlier upload '
    'whose content this matched. Join to it to name the deal the document is on.';

CREATE INDEX IF NOT EXISTS ix_bp_process_monitor_duplicate_of
    ON proc.process_monitor (duplicate_of_id);

COMMIT;
