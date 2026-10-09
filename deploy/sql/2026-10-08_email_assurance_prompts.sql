-- Governed prompts for the email assurance stages: classify, plan, judge.
--
-- NOT APPLIED. Insert only. If a prompt row is missing, the stage reports `unavailable` and
-- does not run from text held in code. Placeholders are replaced by name ({families} etc.);
-- the JSON braces in each template are literal.
--
-- The classifier and planner are instructed that omission is correct. That wording follows the
-- lesson of demand_intake_extract v1->v2 (2026-10-05): a prompt that offers examples gets them
-- copied back, and one that rewards completeness gets values nobody gave.

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc, created_by, last_modified_by)
SELECT 'email_family_classify', 'classification', 'email_drafting_agent',
 $p${"prompt_template": "You decide what kind of procurement email a person's request is asking for. You do not write the email.\n\nThe only families that exist:\n{families}\n\nRules:\n- Choose from that list only. Never name a family that is not listed.\n- confidence is a number from 0 to 1. Most requests are not certain; use under 0.7 when it could reasonably be two of them.\n- candidates: the two most likely families, each with its own confidence.\n- lookup_keys: ONLY identifiers the person actually wrote (a PO number, an invoice number, a contract or quote id), copied exactly, as {\"po_number\": \"...\"}. If none was written, return {}. Never produce an identifier from your own knowledge.\n- user_instruction: what the person asked for, in their own words, copied from the request.\n\nReply with only JSON:\n{\"family_id\": \"...\", \"confidence\": 0.0, \"candidates\": [{\"family_id\": \"...\", \"confidence\": 0.0}], \"lookup_keys\": {}, \"user_instruction\": \"...\"}"}$p$::jsonb,
 'deploy/sql/2026-10-08_email_assurance_prompts.sql', 'deploy/sql/2026-10-08_email_assurance_prompts.sql'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'email_family_classify');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc, created_by, last_modified_by)
SELECT 'email_brief_plan', 'planning', 'email_drafting_agent',
 $p${"prompt_template": "You plan procurement emails. You do not write them. Family: {family}. Tone variables: {tone}. The person's instruction: {instruction}.\n\nFacts verified from the product database (authoritative: never alter, round or extend them): {facts}\nContext items (background, not figures to quote): {context}\n\nReturn only this JSON object. Every value is a plain string unless shown otherwise:\n{\"goal\": \"<one sentence>\",\n \"key_points\": [\"<point>\", \"...\"],\n \"explicit_ask\": \"<the one thing the supplier is asked to do>\",\n \"deadline\": \"<the reply deadline in the person's own words, or null if they gave none>\",\n \"tone_rationale\": \"<one sentence on the tone, as text>\",\n \"risks_to_avoid\": [\"<risk>\", \"...\"],\n \"reasoned\": {\"<judgement name>\": {\"value\": \"<the judgement>\", \"basis\": [\"<fact or context key>\"], \"confidence\": 0.0}},\n \"assumptions\": [\"<anything you assumed>\"]}\n\nRules:\n- reasoned is an object keyed by judgement name; it may be {}. A basis may only name keys listed above; if a judgement has none, give an empty basis and say what you assumed in assumptions.\n- Never invent a deadline. If the person gave none, deadline is null.\n- Do not put any figure, date or reference in the brief that is not in the facts or the person's own words. Do not calculate new figures.\n- Asking the supplier FOR information (a date, a price, a quote) is normal: that information is not missing.\n- Only when the email must STATE to the supplier a value the person referred to but did not give, and the facts do not hold it (for example \"pay on the due date\" with no due date), return exactly {\"missing\": [\"<that value>\"]} instead.\n\nThe person's request follows."}$p$::jsonb,
 'deploy/sql/2026-10-08_email_assurance_prompts.sql', 'deploy/sql/2026-10-08_email_assurance_prompts.sql'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'email_brief_plan');

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc, created_by, last_modified_by)
SELECT 'email_draft_judge', 'evaluation', 'email_drafting_agent',
 $p${"prompt_template": "You score a drafted procurement email. You do not rewrite it.\n\nCriteria (score each from 1 to 5; 5 is best): {rubric}\nThe plan the email was written from: {brief}\nVerified facts: {facts}\n\nRules:\n- Score only what is on the page. A criterion the email does not address scores 1.\n- Whether each reasoned statement follows from its basis is part of the score.\n- Do not reward length.\n\nReply with only JSON: {\"scores\": {\"<criterion>\": 1}, \"rationale\": \"one sentence\"}\n\nThe email follows."}$p$::jsonb,
 'deploy/sql/2026-10-08_email_assurance_prompts.sql', 'deploy/sql/2026-10-08_email_assurance_prompts.sql'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'email_draft_judge');
