-- The column the capture code writes to record what steered a draft (NULL = steering never ran for the row).
--
-- Its own file, in PACK (a), because the capture INSERT names this column: without it every capture would fail, so it cannot wait for
-- pack (b) (tone rules, prompts, steering settings). NOT applied anywhere. Reversible.
BEGIN;
ALTER TABLE email_agent.bp_draft_capture ADD COLUMN IF NOT EXISTS steering JSONB;
COMMENT ON COLUMN email_agent.bp_draft_capture.steering IS
    'What steered the draft: {status, tone:[{variable,value}], style_rule_ids, exemplar_ids, exemplar_scope}. NULL = steering never ran for this row.';
COMMIT;
