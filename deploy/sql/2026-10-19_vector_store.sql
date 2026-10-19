-- The chat/RAG vector store in Postgres (approved 2026-10-19; replaces Qdrant). Applied to bp_testdb first.
--
-- DESIGN
--  * Postgres is the source of truth. This table is DERIVED and fully rebuildable from source rows (python -m src.services.vector_store.rebuild).
--  * ONE shared table for all tenants, with a NON-NULL tenant_id. Not a table per tenant. The store's search() requires a tenant and always
--    filters on it; there is no code path that searches across tenants. NEXT SCALING STEP: partition this table by tenant_id (LIST or HASH)
--    so each tenant's rows and HNSW index are physically separate; the adapter and its queries do not change.
--  * One table PER EMBEDDING MODEL VERSION. A vector's dimension is fixed by its model, so switching models means a NEW table created from this
--    template with the new dimension, backfilled in parallel while this one serves traffic, then cut over by setting VECTOR_PROFILE.
--    embedding_model / embedding_version are stored on every row and CHECKed against the table's profile, and search filters on both, so
--    vectors from different model versions can never be compared.
--  * HNSW on the embedding (cosine: bge-large-en-v1.5 emits normalised vectors and is trained for cosine), plus b-tree indexes on the filters.
--
-- Idempotent. Reversible: 2026-10-19_vector_store_rollback.sql (the data is derived; rollback loses nothing that cannot be rebuilt).
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_vector_chunk_bge_large_v1 (
    vector_id          bigserial PRIMARY KEY,
    tenant_id          text        NOT NULL CHECK (tenant_id <> ''),
    collection         text        NOT NULL,            -- logical collection: documents | uploaded | policy | learning | memory | ...
    source_id          text        NOT NULL,            -- the source row this was derived from
    chunk_idx          integer     NOT NULL DEFAULT 0,
    embedding_model    text        NOT NULL CHECK (embedding_model = 'BAAI/bge-large-en-v1.5'),
    embedding_version  text        NOT NULL CHECK (embedding_version = '1'),
    embedding          vector(1024) NOT NULL,
    content            text,
    document_type      text,
    document_id        text,
    record_id          text,
    supplier_id        text,
    session_id         text,
    product_type       text,
    source_type        text,
    metadata           jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_bp_vector_chunk_bge_large_v1 UNIQUE (tenant_id, collection, source_id, chunk_idx)
);

CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_hnsw
    ON proc.bp_vector_chunk_bge_large_v1 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_tenant_coll ON proc.bp_vector_chunk_bge_large_v1 (tenant_id, collection);
CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_tenant_src  ON proc.bp_vector_chunk_bge_large_v1 (tenant_id, source_id);
CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_doctype     ON proc.bp_vector_chunk_bge_large_v1 (tenant_id, document_type);
CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_supplier    ON proc.bp_vector_chunk_bge_large_v1 (tenant_id, supplier_id);
CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_session     ON proc.bp_vector_chunk_bge_large_v1 (tenant_id, session_id);
CREATE INDEX IF NOT EXISTS ix_bp_vector_chunk_bge_large_v1_docid       ON proc.bp_vector_chunk_bge_large_v1 (tenant_id, document_id);

COMMIT;
