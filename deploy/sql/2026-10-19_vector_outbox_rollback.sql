BEGIN;
DROP TRIGGER IF EXISTS tr_bp_vector_source_enqueue ON proc.bp_vector_source;
DROP FUNCTION IF EXISTS proc.bp_vector_source_enqueue();
DROP TABLE IF EXISTS proc.bp_embedding_job;
DROP TABLE IF EXISTS proc.bp_vector_source;
COMMIT;
