-- Reverse of 2026-10-09_bp_agent_policy_extraction.sql (only for this feature's own objects).
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_agent_policy_source;
ALTER TABLE proc.bp_agent_policy
    DROP COLUMN IF EXISTS source_split,
    DROP COLUMN IF EXISTS source_reference,
    DROP COLUMN IF EXISTS source_document_id;
DROP TABLE IF EXISTS proc.bp_policy_extraction_item;
DROP INDEX IF EXISTS proc.ix_bp_policy_extraction_run_status;
DROP TABLE IF EXISTS proc.bp_policy_extraction_run;
DROP TABLE IF EXISTS proc.bp_policy_document_version;
DROP INDEX IF EXISTS proc.ix_bp_policy_document_match;
DROP TABLE IF EXISTS proc.bp_policy_document;
COMMIT;
