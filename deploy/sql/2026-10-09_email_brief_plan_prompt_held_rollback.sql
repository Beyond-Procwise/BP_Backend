DELETE FROM proc.bp_prompt
 WHERE prompt_name = 'email_brief_plan'
   AND created_by = 'deploy/sql/2026-10-09_email_brief_plan_prompt_held.sql';
