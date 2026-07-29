-- 2026-07-29 Give proc.bp_opportunity a stable identity, and a way to retire
-- findings a later mining run no longer detects.
--
-- The table was keyed on opportunity_id, which the miner assigns from a per-run
-- counter (_next_opportunity_id) during candidate evaluation — so the SAME finding
-- got a different id on every run depending on how many candidates that run walked
-- (one supplier's finding was '3130' in one run and '12123' in the next). Three
-- consequences: re-running mining inserted duplicates instead of updating; nothing
-- was ever retired; and, worst, a colliding id let one run's finding overwrite an
-- unrelated finding from an earlier run AND inherit its lifecycle stage.
--
-- opportunity_ref_id is already the stable, content-derived identity
-- (policy_detector_sourcehash_supplier_item). This makes it the upsert key.
--
-- Additive + idempotent.
BEGIN;

-- retired_at records that a full mining run stopped detecting this finding.
-- stage is set to 'closed' alongside it, so every existing reader (which already
-- excludes 'closed' from open counts) needs no change; retired_at is what keeps
-- "we stopped seeing it" distinguishable from "a human closed it".
ALTER TABLE proc.bp_opportunity
    ADD COLUMN IF NOT EXISTS retired_at TIMESTAMPTZ;

-- Backfill: rows predating opportunity_ref_id cannot be identified by content, so
-- fall back to their own opportunity_id. That keeps them addressable and unique
-- without inventing an identity for them.
UPDATE proc.bp_opportunity
   SET opportunity_ref_id = opportunity_id
 WHERE opportunity_ref_id IS NULL OR btrim(opportunity_ref_id) = '';

-- Collapse any duplicates the old keying already created, keeping the most
-- recently detected row for each identity (and preferring one a human has moved
-- off 'identified', so lifecycle progress is never the row that gets dropped).
WITH ranked AS (
    SELECT opportunity_id,
           row_number() OVER (
               PARTITION BY opportunity_ref_id
               ORDER BY (stage <> 'identified') DESC,
                        detected_on DESC NULLS LAST,
                        updated_at  DESC NULLS LAST
           ) AS rn
      FROM proc.bp_opportunity
)
DELETE FROM proc.bp_opportunity o
 USING ranked r
 WHERE o.opportunity_id = r.opportunity_id
   AND r.rn > 1;

ALTER TABLE proc.bp_opportunity
    ALTER COLUMN opportunity_ref_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_opportunity_ref
    ON proc.bp_opportunity (opportunity_ref_id);

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_retired_at
    ON proc.bp_opportunity (retired_at);

COMMIT;
