-- proc.supplier_response: record where an extracted value came from, and whether a person confirmed it.
--
-- A PRODUCT-TABLE CHANGE, not part of the email_agent pack. NOT APPLIED anywhere. Separate change request.
--
-- Why: the analyser (supplier_interaction_agent._parse_response) takes the first number in an inbound email as the price (a model may
-- override it) and writes it to price / lead_time with no record of how it got there. The email assurance layer reads that column as a
-- Postgres fact. These columns let it tell a CLAIM (read from an email, unconfirmed) from a value a person vouched for.
--
-- Safe to add: eight nullable columns with no default, so it is a metadata-only change on a table that holds 1 row in bp_sqldb (and 7 in
-- bp_testdb). Existing rows are NOT backfilled: their origin cannot be known, and NULL means exactly "origin not recorded". Nothing
-- that reads the table by name is affected; the writer that fills them is best-effort and tolerates the columns being absent.

BEGIN;

ALTER TABLE proc.supplier_response
    ADD COLUMN IF NOT EXISTS extraction_status         TEXT
        CHECK (extraction_status IN ('extracted_unverified', 'confirmed', 'rejected')),
    ADD COLUMN IF NOT EXISTS extraction_method         TEXT,           -- llm | regex_first_number | regex_days
    ADD COLUMN IF NOT EXISTS extraction_model          TEXT,           -- the model that returned the value, when one did
    ADD COLUMN IF NOT EXISTS extraction_prompt_version TEXT,          -- which prompt (the analyser's is inline, so it is named by where it lives)
    ADD COLUMN IF NOT EXISTS extraction_confidence     NUMERIC(4,3),   -- NULL: neither the regex nor the model reports one
    ADD COLUMN IF NOT EXISTS extracted_at              TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS confirmed_by              TEXT,
    ADD COLUMN IF NOT EXISTS confirmed_at              TIMESTAMPTZ;

COMMENT ON COLUMN proc.supplier_response.extraction_status IS
    'NULL = origin never recorded (every row before this change). extracted_unverified = read from the email by software. confirmed = a person vouched for price/lead_time. rejected = a person said it is wrong.';

COMMIT;
