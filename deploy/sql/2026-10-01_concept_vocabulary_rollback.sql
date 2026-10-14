-- Reverses 2026-10-01_concept_vocabulary.sql.
--
-- READ BEFORE RUNNING — this one is NOT a tidy inverse.
--
-- 1. IT DESTROYS HUMAN DECISIONS. DROP TABLE takes confirmed_by and confirmed_at
--    with it for every row a person promoted, rejected or confirmed since the
--    seed went in, together with any concept or document type added by hand
--    (source <> 'seed'). The forward migration seeds with ON CONFLICT DO
--    NOTHING, so re-running it afterwards restores the SEED and nothing else:
--    the human record is gone, not recoverable from code, and seed.py is not a
--    backup of it. Dump proc.bp_concept and proc.bp_document_type first if any
--    of that matters.
--
-- 2. THE ROLLBACKS COMPOSE IN ONE ORDER ONLY. There are two alias migrations
--    with their own rollbacks (2026-10-01_concept_vocabulary_aliases_rollback.sql
--    and 2026-10-01_concept_vocabulary_restored_aliases_rollback.sql) and both
--    UPDATE proc.bp_document_type. Run them BEFORE this file. The reverse order
--    errors on a dropped table, and under ON_ERROR_STOP that aborts the script
--    mid-way; without it, the failure is a message in a log nobody reads.
--    Running them after this file is also pointless even where it does not
--    error: there is no row left to update.
--
-- 3. Nothing else depends on these tables by foreign key, but
--    src/services/concepts/vocabulary.py falls back to the built-in seed when
--    the tables cannot be read, so the application keeps running on a slightly
--    stale vocabulary rather than failing loudly. Expect no alarm.
BEGIN;
DROP TABLE IF EXISTS proc.bp_document_type;
DROP TABLE IF EXISTS proc.bp_concept;
COMMIT;
