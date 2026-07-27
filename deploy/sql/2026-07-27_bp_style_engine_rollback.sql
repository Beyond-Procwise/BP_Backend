BEGIN;
-- deploy/sql/2026-07-27_bp_style_engine_rollback.sql
--
-- Reverses 2026-07-27_bp_style_engine.sql.
--
-- DESTRUCTIVE. Dropping proc.bp_style_profile destroys every compiled style profile and
-- the approval history behind it. Profiles are derived artifacts — they can be recompiled
-- from freshly pasted emails — but the approvals cannot: someone signed off each active
-- version, and that record does not come back. Take a backup of the four bp_style_* tables
-- before running this if any profile has ever been approved.
--
-- The provenance columns on proc.draft_rfq_emails are dropped too. Existing drafts survive
-- (the table itself is untouched); they simply lose the record of which profile wrote them.
--
-- Order matters: drop the dependent columns before the tables they reference.

-- ===================== draft provenance =====================
ALTER TABLE proc.draft_rfq_emails
    DROP COLUMN IF EXISTS style_user_ref,
    DROP COLUMN IF EXISTS style_intent,
    DROP COLUMN IF EXISTS style_mode,
    DROP COLUMN IF EXISTS style_profile_id,
    DROP COLUMN IF EXISTS style_profile_version,
    DROP COLUMN IF EXISTS style_fallback_level,
    DROP COLUMN IF EXISTS style_exemplar_ids,
    DROP COLUMN IF EXISTS style_exemplar_set_hash,
    DROP COLUMN IF EXISTS style_mailbox_binding_id,
    DROP COLUMN IF EXISTS style_message_ids,
    DROP COLUMN IF EXISTS style_retrieved_at,
    DROP COLUMN IF EXISTS style_model_id,
    DROP COLUMN IF EXISTS style_prompt_version,
    DROP COLUMN IF EXISTS external_draft_ref;

-- The index goes with the column it covered, but drop defensively in case the column
-- was removed by hand at some point and the index outlived it.
DROP INDEX IF EXISTS proc.ix_draft_rfq_emails_style_profile;

-- ===================== tables =====================
DROP TABLE IF EXISTS proc.bp_mailbox_binding;
DROP TABLE IF EXISTS proc.bp_style_exemplar;
DROP TABLE IF EXISTS proc.bp_style_profile;
DROP TABLE IF EXISTS proc.bp_style_ingest_staging;
DROP TABLE IF EXISTS proc.bp_style_intent;

-- The `vector` extension is deliberately NOT dropped. Dropping it would cascade into any
-- other pgvector column added since, and an unused extension costs nothing. Remove it by
-- hand if this really is the last pgvector user:
--     DROP EXTENSION IF EXISTS vector;

-- ===================== configuration =====================
DELETE FROM proc.bp_admin_config WHERE config_key = 'style_engine';

COMMIT;
