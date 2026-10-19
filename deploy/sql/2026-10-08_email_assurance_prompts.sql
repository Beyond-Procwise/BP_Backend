-- Governed prompts for the email assurance stages: classify and judge.
--
-- The planner prompt was moved out on 2026-10-09 to 2026-10-09_email_brief_plan_prompt_held.sql, which is in no pack:
-- a 20-request live A/B showed the brief does not improve drafts and doubles the time per draft.
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
SELECT 'email_draft_judge', 'evaluation', 'email_drafting_agent',
 $p${"prompt_template": "You score a drafted procurement email. You do not rewrite it.\n\nCriteria (score each from 1 to 5; 5 is best): {rubric}\nThe plan the email was written from: {brief}\nVerified facts: {facts}\n\nRules:\n- Score only what is on the page. A criterion the email does not address scores 1.\n- Whether each reasoned statement follows from its basis is part of the score.\n- Do not reward length.\n\nReply with only JSON: {\"scores\": {\"<criterion>\": 1}, \"rationale\": \"one sentence\"}\n\nThe email follows."}$p$::jsonb,
 'deploy/sql/2026-10-08_email_assurance_prompts.sql', 'deploy/sql/2026-10-08_email_assurance_prompts.sql'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'email_draft_judge');
