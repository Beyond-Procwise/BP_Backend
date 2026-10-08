-- Drops the provenance columns. Loses the recorded origin of any extracted value; the values themselves are untouched.
BEGIN;
ALTER TABLE proc.supplier_response
    DROP COLUMN IF EXISTS confirmed_at,
    DROP COLUMN IF EXISTS confirmed_by,
    DROP COLUMN IF EXISTS extracted_at,
    DROP COLUMN IF EXISTS extraction_confidence,
    DROP COLUMN IF EXISTS extraction_prompt_version,
    DROP COLUMN IF EXISTS extraction_model,
    DROP COLUMN IF EXISTS extraction_method,
    DROP COLUMN IF EXISTS extraction_status;
COMMIT;
