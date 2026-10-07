-- Report data (design: docs/superpowers/specs/2026-07-13-report-builder-design.md, Stage 2).
-- Additive and idempotent. bp_reports has no CREATE TABLE in version control, so every change
-- to it is ADD COLUMN IF NOT EXISTS. Legacy rows (definition IS NULL) are not touched.

-- Which buyer_id codes a Buyer/Viewer/Approver may see. Admin sees all and needs no row.
-- Appended and revoked, never edited; only an Admin grants (enforced in the API).
CREATE TABLE IF NOT EXISTS proc.bp_user_buyer_scope (
    scope_id    bigserial PRIMARY KEY,
    subject     text        NOT NULL,
    buyer_id    text        NOT NULL,
    granted_by  text        NOT NULL,
    granted_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz,
    revoked_by  text
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_user_buyer_scope_active
    ON proc.bp_user_buyer_scope (subject, buyer_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_bp_user_buyer_scope_subject ON proc.bp_user_buyer_scope (subject);

-- Presentation mode: every activation, deactivation and presentation export. Append-only; the
-- latest activate/deactivate for a session is that session's state.
CREATE TABLE IF NOT EXISTS proc.bp_presentation_log (
    log_id      bigserial PRIMARY KEY,
    subject     text        NOT NULL,
    session_id  text        NOT NULL,
    event       text        NOT NULL CHECK (event IN ('activate','deactivate','export','data')),
    detail      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    expires_at  timestamptz,
    at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_presentation_log_session ON proc.bp_presentation_log (subject, session_id, log_id DESC);

-- A report is fully live or fully presentation. Existing rows are live.
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS data_mode text NOT NULL DEFAULT 'live';
DO $$ BEGIN
    ALTER TABLE proc.bp_reports ADD CONSTRAINT bp_reports_data_mode_chk CHECK (data_mode IN ('live','presentation'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
CREATE INDEX IF NOT EXISTS ix_bp_reports_data_mode ON proc.bp_reports (data_mode);
