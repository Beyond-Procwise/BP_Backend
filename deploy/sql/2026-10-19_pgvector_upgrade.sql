-- Upgrade the pgvector extension to 0.8.0 (approved 2026-10-19, bp_testdb first). 0.8 adds ITERATIVE INDEX SCANS: without them a
-- filtered search (tenant_id = ..., document_type = ...) over an HNSW index can return fewer than k rows, because the index
-- yields its nearest candidates first and the filter then discards most of them. That recall loss is exactly what a shared
-- multi-tenant table suffers, so the vector store sets hnsw.iterative_scan on every search when this version is present.
--
-- NOT SCRIPT-REVERSIBLE: pgvector ships no downgrade path. Existing vector columns and indexes (the two style tables) keep working
-- across the upgrade; take a snapshot first. On RDS the 0.8.0 control file must be available (pg_available_extensions shows it).
DO $$
DECLARE installed text; available text;
BEGIN
    SELECT installed_version, default_version INTO installed, available FROM pg_available_extensions WHERE name = 'vector';
    IF installed IS NULL THEN
        CREATE EXTENSION vector;
    ELSIF installed <> available THEN
        EXECUTE format('ALTER EXTENSION vector UPDATE TO %L', available);
    END IF;
END $$;
