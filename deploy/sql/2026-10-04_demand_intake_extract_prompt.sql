-- 2026-10-04  The governed prompt behind Demand Intake's extraction.
--
-- WHAT IT IS FOR. The intake conversation asks the questions; the model's ONLY job is to read
-- what the requester typed and return values for fields the configuration already named. It is
-- never asked for a question and never asked for an opinion, and every value it returns is
-- validated in code before it reaches the record. That is why the instruction belongs here: it
-- is the one part of the turn a human should be able to change without a release.
--
-- WHY IT IS NOT IN THE BROWSER. The screen used to build this prompt itself and hand it to a
-- host hook. A browser-supplied instruction is not governed — anyone who can call the endpoint
-- could supply any instruction — so the server renders THIS row and nothing else, and the
-- browser now sends only the values to fill it with. If this row is missing the endpoint
-- refuses rather than falling back to whatever the caller sent.
--
-- prompt_type = 'extraction', a NEW value. The nine in use are email_prompt, summary_persona,
-- ask_persona, critique, decomposition, elicitation, justification, message and scoping; none
-- of them means "pull values out of text". `scoping` is the closest and is still wrong — a
-- scoping prompt proposes requirements, which is exactly what this one must never do. The
-- unique index is (prompt_type, prompt_name) on active rows, so the type is part of the row's
-- identity and mislabelling it is not cosmetic.
--
-- prompts_desc is JSONB and holds an OBJECT under `prompt_template` — the key 14 of the 15
-- existing rows use. prompt_linked_agents is the slug of the class that resolves it
-- (DemandIntakeAgent -> demand_intake_agent), which is how resolve_prompt finds it.
--
-- APPLY TO BOTH DATABASES. bp_testdb and bp_sqldb each hold their own copy of proc.bp_prompt
-- (the same 15 rows in both today); one insert does not cover the other.
--
-- prompt_name has no unique constraint on its own, so this is INSERT ... WHERE NOT EXISTS
-- rather than ON CONFLICT. Idempotent: safe to re-run.

BEGIN;

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc,
                            created_by, last_modified_by)
SELECT
    'demand_intake_extract',
    'extraction',
    'demand_intake_agent',
    '{"prompt_template": "You read a procurement request and pull out demand record values. You do not ask questions.\n\nValid categories: {categories}\nFields (path: type or options): {fields}\nAlready known (do not change): {known}\nThe question just asked (may be \"none\"): {asked_field}\nText from the requester: {text}\n\nRules:\n- Only fill a field when the text states it or makes it obvious. Never guess.\n- For option fields use one listed option exactly; for category use one valid category exactly.\n- Dates as YYYY-MM-DD; convert \"March\" or \"in 6 weeks\" to the next such date after {today}.\n- Money as a number in {currency}. A cost centre is a code like \"IT-3300\"; copy it exactly.\n- problem.cur is how things are today; problem.des is how they should be; criteria is a list of\n  measurable outcomes (\"99.95% SLA\", \"≥15% unit-rate reduction\").\n- title: short and plain, under 60 characters.\n- confidence \"high\" when stated, \"medium\" when clearly implied, \"low\" when unsure.\n\nReply with only JSON:\n{\"fields\": {\"path\": {\"value\": \"...\", \"confidence\": \"high|medium|low\"}}}"}'::jsonb,
    'deploy/sql/2026-10-04_demand_intake_extract_prompt.sql',
    'deploy/sql/2026-10-04_demand_intake_extract_prompt.sql'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'demand_intake_extract'
);

COMMIT;
