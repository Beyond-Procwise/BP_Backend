-- A quote's layout, and the logos it can wear.
--
--   * proc.bp_brand_logo keeps every logo uploaded. Each upload is a new row and none
--     is edited in place, so swapping the company's logo never re-dresses a quote that
--     has already gone out. PNG and JPEG only (checked by decoding, in code).
--   * bp_sales_quote.sections is the quote's own layout: its sections in order, each
--     with a kind (what it is), a heading (what the page calls it), the words of any
--     free-text section, and on the letterhead the logo it wears. It is copied from the
--     template when the quote is started and is then the quote's to change. NULL means
--     "the standard layout", which is what every quote written before today has.
--   * A template's layout lives in its body (jsonb), so templates need no new column.
--   * A new governed prompt, sales_quote_sum: asked only when the seller wants something
--     worked out from their own numbers, its one job is to write the sum down
--     ("12000 / 340"); code works it out. The fill prompt is not changed. Sections and
--     the quote's own total, tax and margin are read from the seller's message in code
--     and are in no prompt -- tried there, the model did them badly.
--
-- Additive and idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_brand_logo (
    logo_id      bigserial PRIMARY KEY,
    logo_name    text        NOT NULL,
    content_type text        NOT NULL,
    sha256       text        NOT NULL,
    size_bytes   integer     NOT NULL,
    width        integer     NOT NULL,
    height       integer     NOT NULL,
    image        bytea       NOT NULL,
    status       text        NOT NULL DEFAULT 'active',   -- active | removed
    created_by   text,
    created_date timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_brand_logo_status ON proc.bp_brand_logo (status);

ALTER TABLE proc.bp_sales_quote
    ADD COLUMN IF NOT EXISTS sections jsonb;

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc,
                            created_by, last_modified_by)
SELECT
    'sales_quote_sum',
    'extraction',
    'quote_drafting',
    jsonb_build_object('prompt_template',
'A seller working on a customer quote asks for something to be worked out. Write the sum down. Do NOT work it out: a calculator will.

The quote''s lines (ids and what each is): {quote}
What the seller says: {message}

Give:
- "expression": the sum, using ONLY numbers the seller typed and + - * / ( ). Write a percentage as 4%. "18k" is 18000. No words, no units, no currency signs, no equals sign.
  Examples: a budget of 12000 across 340 seats, price per seat -> 12000 / 340
            4,200 a month for 24 months, over the whole term -> 4200 * 24
            200 each less 4% -> 200 * (1 - 4%)
            340 seats at 200 plus 18k onboarding -> 340 * 200 + 18000
- "label": a few words saying what the answer is.
- "line_id" and "into": ONLY if the seller asks for the answer to be used on a line. "line_id" is that line''s id from the quote above and "into" is qty, unit_price or cost. Otherwise both are empty strings.

If the seller is not asking for a sum on their own numbers (for example they only ask for the quote''s total or its margin), "expression" is an empty string.

Reply with only JSON: {"expression": "", "label": "", "line_id": "", "into": ""}'),
    'deploy/sql/2026-10-13_bp_quote_sections_logo.sql',
    'deploy/sql/2026-10-13_bp_quote_sections_logo.sql'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'sales_quote_sum'
);

COMMIT;
