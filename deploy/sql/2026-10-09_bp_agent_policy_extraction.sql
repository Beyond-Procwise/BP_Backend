-- Agent policy governance stage 2: extraction agent storage. Additive and idempotent.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_policy_document (
    document_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title         TEXT NOT NULL,                 -- from the first upload's filename, editable later
    match_name    TEXT NOT NULL,                 -- normalised filename used to recognise a revision
    latest_version INTEGER NOT NULL DEFAULT 1,
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_document_match ON proc.bp_policy_document (match_name);

CREATE TABLE IF NOT EXISTS proc.bp_policy_document_version (
    document_id   BIGINT NOT NULL REFERENCES proc.bp_policy_document (document_id),
    version       INTEGER NOT NULL CHECK (version >= 1),
    filename      TEXT NOT NULL,
    s3_key        TEXT NOT NULL,
    byte_size     BIGINT NOT NULL CHECK (byte_size > 0),
    content_hash  TEXT NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    parsed_text   TEXT,                          -- NULL until the first run parses it
    parsed_at     TIMESTAMPTZ,
    uploaded_by   TEXT NOT NULL,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, version),
    CONSTRAINT ux_bp_policy_document_version_hash UNIQUE (document_id, content_hash)
);

CREATE TABLE IF NOT EXISTS proc.bp_policy_extraction_run (
    run_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('extract','fix')),
    status        TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
    request       JSONB NOT NULL,                -- extract: {"documents":[{"documentId","version"}]}; fix: {"policyKey","baseVersion","form"}
    owner         TEXT,                          -- process that claimed it
    heartbeat_at  TIMESTAMPTZ,
    counts        JSONB NOT NULL DEFAULT '{}',
    error         TEXT,
    started_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_bp_policy_extraction_run_status ON proc.bp_policy_extraction_run (status);

CREATE TABLE IF NOT EXISTS proc.bp_policy_extraction_item (
    run_id        BIGINT NOT NULL REFERENCES proc.bp_policy_extraction_run (run_id),
    seq           INTEGER NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('policy','not_enforceable','proposed_retire','fix','error','note')),
    document_id   BIGINT,
    document_version INTEGER,
    reference     TEXT,
    payload       JSONB NOT NULL,
    policy_key    TEXT,
    decision      TEXT CHECK (decision IS NULL OR decision IN ('new','changed','unchanged','proposed_retire')),
    saved_version INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);

ALTER TABLE proc.bp_agent_policy
    ADD COLUMN IF NOT EXISTS source_document_id BIGINT REFERENCES proc.bp_policy_document (document_id),
    ADD COLUMN IF NOT EXISTS source_reference TEXT,
    ADD COLUMN IF NOT EXISTS source_split TEXT;   -- distinguishes tiered policies from one clause (e.g. the outcome)
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_source ON proc.bp_agent_policy (source_document_id, source_reference);
COMMIT;
