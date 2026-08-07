-- scripts/migrations/2026-08-07-mail-intake.sql
--
-- Email attachment ingestion. Attachments arriving by email are dropped today:
-- email_ingest_lambda.py handles none, imap_supplier_response_watcher.py stores
-- attachment metadata without the bytes, and email_watcher.py skips any MIME
-- part with an attachment disposition.
--
-- Tenancy: proc has zero tenant_id columns, so binding_id is the isolation
-- scope and the binding OWNER (bp_mailbox_binding.user_ref) is the dedup scope.
-- See spec §5.3 before adding a tenant column here.
--
-- bp_ prefix per the project convention; ix_bp_<table>_<col> for indexes.

ALTER TABLE proc.bp_mailbox_binding
    ADD COLUMN IF NOT EXISTS folder_scope         TEXT;
ALTER TABLE proc.bp_mailbox_binding
    ADD COLUMN IF NOT EXISTS poll_interval_seconds INTEGER;
ALTER TABLE proc.bp_mailbox_binding
    ADD COLUMN IF NOT EXISTS last_sync_at         TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS proc.bp_ingest_bundle (
    bundle_id           UUID PRIMARY KEY,
    binding_id          TEXT NOT NULL,
    owner_ref           TEXT NOT NULL,
    provider_message_id TEXT,
    rfc_message_id      TEXT,
    thread_id           TEXT,
    sender_address      TEXT,
    sender_display_name TEXT,
    sender_domain       TEXT,
    recipients          JSONB,
    subject             TEXT,
    received_at         TIMESTAMPTZ,
    body_text           TEXT,
    body_html           TEXT,
    raw_mime_key        TEXT NOT NULL,
    state               TEXT NOT NULL CHECK (state IN (
                            'received','parsed','authenticated','bound',
                            'complete','quarantined')),
    quarantine_reason   TEXT CHECK (quarantine_reason IS NULL OR quarantine_reason IN (
                            'malformed_mime','unknown_type','encrypted','macro',
                            'archive','executable','oversize','type_mismatch',
                            'hidden_text','unresolvable_reference')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_ingest_bundle_binding_id
    ON proc.bp_ingest_bundle (binding_id);
CREATE INDEX IF NOT EXISTS ix_bp_ingest_bundle_state
    ON proc.bp_ingest_bundle (state);
CREATE INDEX IF NOT EXISTS ix_bp_ingest_bundle_sender_domain
    ON proc.bp_ingest_bundle (sender_domain);
-- Idempotency for re-delivered provider events and IMAP re-polls.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_ingest_bundle_binding_provider_msg
    ON proc.bp_ingest_bundle (binding_id, provider_message_id)
    WHERE provider_message_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS proc.bp_ingest_artifact (
    artifact_id             UUID PRIMARY KEY,
    bundle_id               UUID NOT NULL REFERENCES proc.bp_ingest_bundle (bundle_id),
    binding_id              TEXT NOT NULL,
    owner_ref               TEXT NOT NULL,
    declared_filename       TEXT,
    declared_mime           TEXT,
    detected_mime           TEXT,
    size_bytes              BIGINT,
    sha256                  TEXT,
    object_key              TEXT,
    nesting_depth           INTEGER NOT NULL DEFAULT 0,
    parent_artifact_id      UUID REFERENCES proc.bp_ingest_artifact (artifact_id),
    disposition             TEXT,
    content_id              TEXT,
    duplicate_of_artifact_id UUID REFERENCES proc.bp_ingest_artifact (artifact_id),
    state                   TEXT NOT NULL CHECK (state IN (
                                'detached','typed','gated','stored',
                                'ready_for_extraction','extraction_complete',
                                'quarantined','duplicate')),
    quarantine_reason       TEXT CHECK (quarantine_reason IS NULL OR quarantine_reason IN (
                                'malformed_mime','unknown_type','encrypted','macro',
                                'archive','executable','oversize','type_mismatch',
                                'hidden_text','unresolvable_reference')),
    quarantine_detail       JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_ingest_artifact_bundle_id
    ON proc.bp_ingest_artifact (bundle_id);
-- The emit queue: stage 10 hands off by selecting on this.
CREATE INDEX IF NOT EXISTS ix_bp_ingest_artifact_state
    ON proc.bp_ingest_artifact (state);
-- Dedup scope is the binding owner (spec §5.3), so one document arriving at two
-- mailboxes of the same customer still resolves to one stored object.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_ingest_artifact_owner_sha
    ON proc.bp_ingest_artifact (owner_ref, sha256)
    WHERE sha256 IS NOT NULL AND duplicate_of_artifact_id IS NULL;

CREATE TABLE IF NOT EXISTS proc.bp_ingest_sender_auth (
    bundle_id           UUID PRIMARY KEY REFERENCES proc.bp_ingest_bundle (bundle_id),
    spf                 TEXT,
    dkim                TEXT,
    dmarc               TEXT,
    envelope_from       TEXT,
    header_from         TEXT,
    alignment           TEXT,
    asserted_by         TEXT,
    supplier_match_state TEXT CHECK (supplier_match_state IN (
                            'matched','unmatched','ambiguous')),
    matched_supplier_id TEXT,
    candidate_supplier_ids JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS proc.bp_ingest_arrival (
    arrival_id      BIGSERIAL PRIMARY KEY,
    artifact_id     UUID NOT NULL REFERENCES proc.bp_ingest_artifact (artifact_id),
    bundle_id       UUID NOT NULL REFERENCES proc.bp_ingest_bundle (bundle_id),
    arrived_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_ingest_arrival_artifact_id
    ON proc.bp_ingest_arrival (artifact_id);

-- Append-only. Reprocessing adds a row; it never updates one.
CREATE TABLE IF NOT EXISTS proc.bp_artifact_derivation (
    derivation_id       BIGSERIAL PRIMARY KEY,
    artifact_id         UUID NOT NULL REFERENCES proc.bp_ingest_artifact (artifact_id),
    pipeline_version    TEXT,
    extraction_run_ref  TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    outcome             TEXT
);

CREATE INDEX IF NOT EXISTS ix_bp_artifact_derivation_artifact_id
    ON proc.bp_artifact_derivation (artifact_id);
