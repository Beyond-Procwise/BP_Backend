-- Agent policy governance stage 2, final review: two lookup indexes. Additive and idempotent.
--   ix_bp_policy_extraction_item_policy_key: the extraction run reads each policy's last agent-saved
--     version from its items (extraction_run._existing), once per existing policy.
--   ix_bp_policy_document_version_s3_key: an asNew register retry finds the version it already
--     made by its S3 key (documents._version_by_key).
BEGIN;
CREATE INDEX IF NOT EXISTS ix_bp_policy_extraction_item_policy_key
    ON proc.bp_policy_extraction_item (policy_key);
CREATE INDEX IF NOT EXISTS ix_bp_policy_document_version_s3_key
    ON proc.bp_policy_document_version (s3_key);
COMMIT;
