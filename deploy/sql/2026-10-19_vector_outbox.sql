-- The embedding pipeline's source and outbox tables (approved 2026-10-19; bp_testdb first).
--
--  bp_vector_source : Postgres-side source rows for content that has no other home (uploaded documents, policy text, learning
--                     records, conversation memory). Document text for the `documents` collection is NOT copied here: the source of truth
--                     there is the _raw row's parser_snapshot, read by its loader.
--  bp_embedding_job : the OUTBOX. A write to a source enqueues a job in the SAME transaction; a worker embeds and upserts with retries.
--
-- Triggers on bp_vector_source (our own table) enqueue automatically, so no writer can forget: insert/update -> an upsert job, delete -> a
-- delete job (the vectors go with the row). A pending job for the same source is coalesced (one pending job per source/action/profile).
-- The `documents` collection is enqueued from write_raw, in the same transaction as the raw row (src/services/extraction/persistence.py).
-- Idempotent. Reversible: 2026-10-19_vector_outbox_rollback.sql (derived data; the vectors can be rebuilt).
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_vector_source (
    tenant_id   text        NOT NULL CHECK (tenant_id <> ''),
    collection  text        NOT NULL,
    source_id   text        NOT NULL,
    content     text        NOT NULL,
    metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, collection, source_id)
);

CREATE TABLE IF NOT EXISTS proc.bp_embedding_job (
    job_id           bigserial PRIMARY KEY,
    tenant_id        text        NOT NULL CHECK (tenant_id <> ''),
    collection       text        NOT NULL,
    source_id        text        NOT NULL,
    action           text        NOT NULL CHECK (action IN ('upsert', 'delete')),
    profile          text,                                   -- NULL = the active profile; a name = backfill that profile's table
    status           text        NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    attempts         integer     NOT NULL DEFAULT 0,
    max_attempts     integer     NOT NULL DEFAULT 5,
    next_attempt_at  timestamptz NOT NULL DEFAULT now(),
    locked_at        timestamptz,
    last_error       text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    done_at          timestamptz
);
-- At most one PENDING job per source/action/profile: a burst of writes to one source is one embedding, not many.
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_embedding_job_pending
    ON proc.bp_embedding_job (tenant_id, collection, source_id, action, (COALESCE(profile, ''))) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS ix_bp_embedding_job_due    ON proc.bp_embedding_job (next_attempt_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS ix_bp_embedding_job_status ON proc.bp_embedding_job (status, updated_at);

CREATE OR REPLACE FUNCTION proc.bp_vector_source_enqueue() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO proc.bp_embedding_job (tenant_id, collection, source_id, action) VALUES (OLD.tenant_id, OLD.collection, OLD.source_id, 'delete')
            ON CONFLICT (tenant_id, collection, source_id, action, (COALESCE(profile, ''))) WHERE status = 'pending' DO NOTHING;
        RETURN OLD;
    END IF;
    IF TG_OP = 'INSERT' OR NEW.content IS DISTINCT FROM OLD.content OR NEW.metadata IS DISTINCT FROM OLD.metadata THEN
        INSERT INTO proc.bp_embedding_job (tenant_id, collection, source_id, action) VALUES (NEW.tenant_id, NEW.collection, NEW.source_id, 'upsert')
            ON CONFLICT (tenant_id, collection, source_id, action, (COALESCE(profile, ''))) WHERE status = 'pending' DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_bp_vector_source_enqueue ON proc.bp_vector_source;
CREATE TRIGGER tr_bp_vector_source_enqueue AFTER INSERT OR UPDATE OR DELETE ON proc.bp_vector_source
    FOR EACH ROW EXECUTE FUNCTION proc.bp_vector_source_enqueue();

COMMIT;
