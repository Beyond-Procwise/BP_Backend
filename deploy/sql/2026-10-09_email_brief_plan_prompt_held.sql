-- HELD: the planner prompt, in NO deployment pack. Do not apply.
--
-- Moved out of 2026-10-08_email_assurance_prompts.sql on 2026-10-09 after a live A/B (20 invented requests, AgentNick:unified,
-- quiet GPU): with the brief 18/20 drafts passed every check, judge mean 4.53, 40.6 s per draft; without it 19/20, 4.51, 20.1 s.
-- Per request the brief scored higher on 7, lower on 9, equal on 4. Without this row the planner stage reports `unavailable`
-- and the writer works from the request alone, which is what the running service already does.
--
-- Kept, not deleted, so the planner code path stays tested: the eval harness loads it (evals/email/db.py HELD); the
-- rehearsal and the packs do not. Rollback: _rollback.sql.

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc, created_by, last_modified_by)
SELECT 'email_brief_plan', 'planning', 'email_drafting_agent',
 $p${"prompt_template": "You plan procurement emails. You do not write them. Family: {family}. Tone variables: {tone}. The person's instruction: {instruction}.\n\nFacts verified from the product database (authoritative: never alter, round or extend them): {facts}\nContext items (background, not figures to quote): {context}\n\nReturn only this JSON object. Every value is a plain string unless shown otherwise:\n{\"goal\": \"<one sentence>\",\n \"key_points\": [\"<point>\", \"...\"],\n \"explicit_ask\": \"<the one thing the supplier is asked to do>\",\n \"deadline\": \"<the reply deadline in the person's own words, or null if they gave none>\",\n \"tone_rationale\": \"<one sentence on the tone, as text>\",\n \"risks_to_avoid\": [\"<risk>\", \"...\"],\n \"reasoned\": {\"<judgement name>\": {\"value\": \"<the judgement>\", \"basis\": [\"<fact or context key>\"], \"confidence\": 0.0}},\n \"assumptions\": [\"<anything you assumed>\"]}\n\nRules:\n- reasoned is an object keyed by judgement name; it may be {}. A basis may only name keys listed above; if a judgement has none, give an empty basis and say what you assumed in assumptions.\n- Never invent a deadline. If the person gave none, deadline is null.\n- Do not put any figure, date or reference in the brief that is not in the facts or the person's own words. Do not calculate new figures.\n- Asking the supplier FOR information (a date, a price, a quote) is normal: that information is not missing.\n- Only when the email must STATE to the supplier a value the person referred to but did not give, and the facts do not hold it (for example \"pay on the due date\" with no due date), return exactly {\"missing\": [\"<that value>\"]} instead.\n\nThe person's request follows."}$p$::jsonb,
 'deploy/sql/2026-10-09_email_brief_plan_prompt_held.sql', 'deploy/sql/2026-10-09_email_brief_plan_prompt_held.sql'
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'email_brief_plan');
