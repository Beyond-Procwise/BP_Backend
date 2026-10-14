-- scripts/migrations/2026-10-14-mail-intake-dedup-index.sql
--
-- One READ original per (owner, content), not one ROW.
--
-- ix_bp_ingest_artifact_owner_sha covered every row with a hash and no
-- duplicate_of. insert_artifact writes the hash when it creates the row, at
-- DETACHED, before the pipeline's own dedup step runs — so the second arrival of
-- any file raised UniqueViolation at insert and failed its whole message. The
-- pipeline's mark_duplicate path was unreachable on Postgres; the unit tests ran
-- against a fake cursor with no index. Found by the first live relay run,
-- 2026-10-14. It also let a QUARANTINED copy hold the slot, so a clean re-send of
-- a once-quarantined file could never be read.
--
-- The index now covers exactly the states repo.ORIGINAL_STATES names, which is
-- also what find_existing_artifact matches: the copy that was handed to
-- extraction. In-flight and quarantined rows hold no slot.
--
-- Idempotent.

DROP INDEX IF EXISTS proc.ix_bp_ingest_artifact_owner_sha;

CREATE UNIQUE INDEX ix_bp_ingest_artifact_owner_sha
    ON proc.bp_ingest_artifact (owner_ref, sha256)
    WHERE sha256 IS NOT NULL
      AND duplicate_of_artifact_id IS NULL
      AND state IN ('ready_for_extraction', 'extraction_complete');
