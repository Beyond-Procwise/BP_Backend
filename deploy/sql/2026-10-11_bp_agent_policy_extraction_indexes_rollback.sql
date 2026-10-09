-- Reverse of 2026-10-11_bp_agent_policy_extraction_indexes.sql (only this file's two indexes).
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_policy_document_version_s3_key;
DROP INDEX IF EXISTS proc.ix_bp_policy_extraction_item_policy_key;
COMMIT;
