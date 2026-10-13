-- 2026-10-13  The governed prompt behind the Quote builder's agent bar (quote_drafting).
--
-- WHAT IT IS FOR. The seller types an instruction ("apply 40% markup to all items"); the
-- model's ONLY job is to say which of six kinds of change that is. It is given no cost
-- and no price and has no field to answer one in. Every figure in the proposal the
-- seller then sees is worked out in code from figures already on the quote
-- (src/services/sell_side/quote_agent.py), and nothing reaches the quote until the
-- seller presses Apply.
--
-- prompt_type = 'extraction': it pulls a structured request out of text, the same job
-- as demand_intake_extract. prompts_desc is JSONB holding an object under
-- `prompt_template`. A missing row fails the turn (HTTP 503); no text in code stands in.
--
-- APPLY TO BOTH DATABASES (bp_testdb and bp_sqldb each hold their own proc.bp_prompt).
-- Idempotent: INSERT ... WHERE NOT EXISTS.

BEGIN;

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc,
                            created_by, last_modified_by)
SELECT
    'sales_quote_drafting',
    'extraction',
    'quote_drafting',
    jsonb_build_object('prompt_template',
'You help a seller edit a customer quote. You do not edit it yourself and you never state a price or a cost. You read the seller''s instruction and say which changes it asks for.

Today is {today}.
The quote (line ids, what each line is, and whether it has a cost): {quote}
The seller''s instruction: {message}

The changes you may name, as "op":
- markup_all: set one markup percent on every line. Give "percent".
- markup_line: set a markup percent on one line. Give "line_id" and "percent".
- discount_all: set one discount percent on every line. Give "percent".
- discount_line: set a discount percent on one line. Give "line_id" and "percent".
- valid_until: change the date the quote is valid until. Give "date" as YYYY-MM-DD; turn "30 days" or "end of month" into a date counted from today.
- add_line: add a new line. Give "description", and "qty", "unit" and "tag" (product, service or one-off) only if the seller said them.

Rules:
- Use only a line_id from the quote above. If you cannot tell which line is meant, name no change.
- Never give an amount of money. If the instruction is to set a price or a cost directly, name no change.
- If the instruction asks for anything not in the list, or you are unsure, return an empty "intents" list and say why in "refused".
- When you do name changes, set "refused" to an empty string.

Reply with only JSON: {"refused": "", "intents": [{"op": "...", ...}]}'),
    'deploy/sql/2026-10-13_sales_quote_drafting_prompt.sql',
    'deploy/sql/2026-10-13_sales_quote_drafting_prompt.sql'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'sales_quote_drafting'
);

COMMIT;
