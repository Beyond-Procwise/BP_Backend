-- 1.20 (a): the egress audit row carries the calling agent and the SHADOW decision.
-- Idempotent; adds three nullable columns, drops nothing. The application's own
-- ensure_schema() runs the same statements at startup, so this file is for a database
-- the application does not start against (a DBA-run path). Rollback: the three columns
-- can be dropped; nothing reads them yet.
ALTER TABLE proc.bp_egress_event ADD COLUMN IF NOT EXISTS agent TEXT;
ALTER TABLE proc.bp_egress_event ADD COLUMN IF NOT EXISTS shadow_decision TEXT;
ALTER TABLE proc.bp_egress_event ADD COLUMN IF NOT EXISTS shadow_reason TEXT;
