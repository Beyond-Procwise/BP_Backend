-- Gap-free customer document numbers (QDA B6).
--
-- A quote's reference today is SQ-<date>-<row id>: made from a sequence, so a rolled-back
-- draft leaves a hole, and it exists from the first save. A number a customer is given has
-- to be the next one, with none missing, and it is given when the quote is issued.
--
--   proc.bp_doc_number_counter   one row per document type per year, holding the last
--                                number handed out. It is incremented in the same
--                                transaction as the status change that issues the
--                                document, so a failed issue hands its number back.
--   bp_sales_quote.quote_number  the number that quote was given. Empty until issue.
--                                quote_ref stays as the internal draft reference.
--
-- Existing quotes are not numbered by this: a number is minted at issue, never afterwards.
--
-- Also sets tenant_id on quotes that have none. UNIQUE (tenant_id, quote_ref) has never
-- been enforced, because every quote was written with a NULL tenant and NULLs are distinct.
-- There is one tenant; it is called 'default' here as it is on a Fact Pack.
--
-- Idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_doc_number_counter (
    doc_type            text    NOT NULL,
    year                integer NOT NULL,
    last_value          bigint  NOT NULL CHECK (last_value >= 1),
    last_modified_date  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_type, year)
);

ALTER TABLE proc.bp_sales_quote ADD COLUMN IF NOT EXISTS quote_number text;

CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_sales_quote_quote_number
    ON proc.bp_sales_quote (quote_number) WHERE quote_number IS NOT NULL;

UPDATE proc.bp_sales_quote SET tenant_id = 'default' WHERE tenant_id IS NULL;

COMMIT;
