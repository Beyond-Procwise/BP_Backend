BEGIN;
ALTER TABLE email_agent.bp_draft_outcome
    DROP COLUMN IF EXISTS reviewed_by, DROP COLUMN IF EXISTS sent_by,
    DROP COLUMN IF EXISTS abandoned_by, DROP COLUMN IF EXISTS abandon_reason;
ALTER TABLE email_agent.bp_draft_capture
    DROP COLUMN IF EXISTS initiated_by, DROP COLUMN IF EXISTS initiated_by_kind, DROP COLUMN IF EXISTS family_source,
    DROP COLUMN IF EXISTS classification, DROP COLUMN IF EXISTS clarification, DROP COLUMN IF EXISTS lookup_keys,
    DROP COLUMN IF EXISTS user_instruction, DROP COLUMN IF EXISTS tone_variables, DROP COLUMN IF EXISTS tone_sources,
    DROP COLUMN IF EXISTS exemplar_ids, DROP COLUMN IF EXISTS exemplar_scope, DROP COLUMN IF EXISTS brief,
    DROP COLUMN IF EXISTS assumption_items, DROP COLUMN IF EXISTS assumptions_resolution, DROP COLUMN IF EXISTS judge,
    DROP COLUMN IF EXISTS authority, DROP COLUMN IF EXISTS stage_status, DROP COLUMN IF EXISTS ready,
    DROP COLUMN IF EXISTS needs_redraft, DROP COLUMN IF EXISTS ready_at,
    ADD COLUMN IF NOT EXISTS user_id TEXT;
COMMIT;
