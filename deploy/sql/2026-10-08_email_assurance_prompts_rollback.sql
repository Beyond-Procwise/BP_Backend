DELETE FROM proc.bp_prompt
 WHERE prompt_name IN ('email_family_classify', 'email_brief_plan', 'email_draft_judge')
   AND created_by = 'deploy/sql/2026-10-08_email_assurance_prompts.sql';
