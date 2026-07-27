BEGIN;
-- deploy/sql/2026-07-27_bp_prompt_style_drafting_rollback.sql
--
-- Reverses 2026-07-27_bp_prompt_style_drafting.sql.
--
-- Drafting keeps working: services/style/drafting.py falls back to a byte-identical
-- constant. Drafts written after this will record prompt version 0 rather than a governed
-- version, which is correct — there is no longer a governed version they ran under.
--
-- Deactivates rather than deletes, so any draft already citing this prompt version can
-- still be traced back to the wording it was written under.

UPDATE proc.bp_prompt
   SET prompts_status     = 0,
       last_modified_date = now(),
       last_modified_by   = 'style-drafting-rollback'
 WHERE prompt_name = 'style_draft_system';

COMMIT;
