-- 2026-10-05  Demand Intake extraction, v2: stop the instruction answering for the requester.
--
-- WHY. The v1 row (2026-10-04_demand_intake_extract_prompt.sql) had never been run against a
-- loaded model. It was, on 2026-10-05, against a resident AgentNick:unified. One request --
-- "SD-WAN connectivity for 42 UK branch sites, live by 31 March 2027. Budget is about GBP240,000
-- over three years on cost centre CC-4120. Today the MPLS circuits cost us GBP95k a year and drop
-- out weekly." -- came back with THIRTY fields, every single one of them claiming "high"
-- confidence. Thirteen of the thirty are in no part of that request:
--
--   - criteria = ['99.95% SLA', '<=1 weekly outage', '>=15% unit-rate reduction'] -- two of the
--     three COPIED VERBATIM out of v1's own wording, which offered them as what a measurable
--     outcome looks like,
--   - finance.saving = 150000, which is neither stated nor arithmetic: three years at GBP95k is
--     GBP285k against GBP240k, so GBP45k, not GBP150k,
--   - finance.phasing = 'Year 1: GBP80k, Year 2: GBP80k, Year 3: GBP80k', benefit.target,
--     benefit.owner, pillar and alignment -- all invented,
--   - intake.existing_contract = 'No', where the request says MPLS circuits are running today.
--
-- Values nobody asked for, on a record that decides an approval route, every one of them flying
-- the highest confidence the scale has. The uniform "high" is the worse half: the browser's
-- validation lets a high-confidence reading correct an earlier machine one, so a confidence that
-- is always high is a model that always overwrites.
--
-- WHAT CHANGED. Omission is now the first rule and is stated as the correct answer rather than a
-- fallback; the fields that are judgement rather than evidence (saving, phasing, targets, owner,
-- pillar, alignment) are named as things never to work out; the illustrative values v1 quoted are
-- gone, so there is nothing to copy; and confidence is described as a thing most answers do not
-- earn. The code guard in src/agents/demand_intake_agent.py drops a value lifted from whatever
-- this row quotes, so an admin who adds an example back does not reopen the hole.
--
-- APPLY TO BOTH DATABASES: bp_testdb and bp_sqldb each hold their own proc.bp_prompt.
-- Idempotent: the UPDATE is by name and rewrites the template to the same text on a re-run, and
-- the IS DISTINCT FROM guard means a second run changes nought rows and does not bump the version.

BEGIN;

UPDATE proc.bp_prompt
   SET prompts_desc = $prompt${"prompt_template": "You read a procurement request and pull out demand record values. You do not ask questions.\n\nValid categories: {categories}\nFields (path: type or options): {fields}\nAlready known (do not change): {known}\nThe question just asked (may be \"none\"): {asked_field}\nText from the requester: {text}\n\nRules:\n- OMIT a field unless the text states it or makes it plain. Returning three fields is a correct\n  answer to a request that states three things; filling every field in the list is wrong.\n- Never work out a value the requester did not give. Do not calculate a saving, split a budget\n  into years, set a target, name an owner, or choose a strategic pillar or alignment. A request\n  does not contain those, and a figure you compute is not something the requester asked for.\n- Do not copy an example out of these rules. Every value must come from the text above.\n- For option fields use one listed option exactly; for category use one valid category exactly.\n- Dates as YYYY-MM-DD; turn a month name or a relative period into the next such date\n  after {today}.\n- Money as a number in {currency}, digits only. A cost centre is a short code written in the\n  text; copy it exactly as it appears.\n- problem.cur is how things are today; problem.des is how they should be; criteria is a list of\n  measurable outcomes, in the requester's own figures and words.\n- title: short and plain, under 60 characters.\n- confidence: \"high\" only where the text says it in words, \"medium\" where the text clearly\n  implies it, \"low\" otherwise. Most requests do not state most fields, so most answers are\n  short and not everything in them is high.\n\nReply with only JSON, and leave out every field the text does not support:\n{\"fields\": {\"path\": {\"value\": \"...\", \"confidence\": \"high|medium|low\"}}}"}$prompt$::jsonb,
       version = COALESCE(version, 1) + 1,
       last_modified_date = now(),
       last_modified_by = 'deploy/sql/2026-10-05_demand_intake_extract_prompt_v2.sql'
 WHERE prompt_name = 'demand_intake_extract'
   AND prompt_type = 'extraction'
   AND prompts_desc->>'prompt_template' IS DISTINCT FROM $tpl$You read a procurement request and pull out demand record values. You do not ask questions.

Valid categories: {categories}
Fields (path: type or options): {fields}
Already known (do not change): {known}
The question just asked (may be "none"): {asked_field}
Text from the requester: {text}

Rules:
- OMIT a field unless the text states it or makes it plain. Returning three fields is a correct
  answer to a request that states three things; filling every field in the list is wrong.
- Never work out a value the requester did not give. Do not calculate a saving, split a budget
  into years, set a target, name an owner, or choose a strategic pillar or alignment. A request
  does not contain those, and a figure you compute is not something the requester asked for.
- Do not copy an example out of these rules. Every value must come from the text above.
- For option fields use one listed option exactly; for category use one valid category exactly.
- Dates as YYYY-MM-DD; turn a month name or a relative period into the next such date
  after {today}.
- Money as a number in {currency}, digits only. A cost centre is a short code written in the
  text; copy it exactly as it appears.
- problem.cur is how things are today; problem.des is how they should be; criteria is a list of
  measurable outcomes, in the requester's own figures and words.
- title: short and plain, under 60 characters.
- confidence: "high" only where the text says it in words, "medium" where the text clearly
  implies it, "low" otherwise. Most requests do not state most fields, so most answers are
  short and not everything in them is high.

Reply with only JSON, and leave out every field the text does not support:
{"fields": {"path": {"value": "...", "confidence": "high|medium|low"}}}$tpl$;

COMMIT;
