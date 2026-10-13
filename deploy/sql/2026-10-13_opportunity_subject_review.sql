-- 2026-10-13 An opportunity is identified by WHAT it is about, not by which
-- documents happened to support it on the day it was found.
--
-- opportunity_ref_id carries a fingerprint of the supporting documents
-- (policy_detector_sourcehash_supplier_item). One more invoice against the same
-- order changed the fingerprint, so the same opportunity came back as a brand-new
-- row at 'identified' -- next to the one a person had already rejected, with no
-- link between the two.
--
-- subject_key is the identity without the documents: the kind of saving, the
-- supplier, and the item (or, with no item, the deal; with neither, the category).
-- The documents stay on the row as evidence. opportunity_ref_id is left alone:
-- proc.bp_finding_fact and the critique tables address rows by it.
--
-- When the evidence under a REJECTED opportunity changes, the row is not
-- re-raised; review_requested_at is set and review_detail says what changed, so a
-- person decides whether the rejection still stands. That needs one new legal
-- move: rejected -> identified.
--
-- The expression below must match subject_key() in src/services/opportunity_store.py;
-- tests/services/test_opportunity_subject_identity.py holds the two to each other.
--
-- Additive + idempotent. Reversible: 2026-10-13_opportunity_subject_review_rollback.sql.
BEGIN;

ALTER TABLE proc.bp_opportunity
    ADD COLUMN IF NOT EXISTS subject_key         TEXT,
    ADD COLUMN IF NOT EXISTS review_requested_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS review_detail       JSONB;

-- One row per subject gets the key. Where the old keying already left several rows
-- for one subject, the one a person has acted on wins (so a rejection is never the
-- row left without an identity), then the most recently detected. The others keep
-- a NULL key and stay addressable by opportunity_ref_id, exactly as before.
WITH keyed AS (
    SELECT opportunity_id,
           CASE
             WHEN coalesce(btrim(supplier_id), '') = ''
              AND coalesce(btrim(item_id), '') = ''
              AND coalesce(btrim(deal_id), '') = ''
              AND coalesce(btrim(category_id), '') = '' THEN NULL
             ELSE lower(
                    coalesce(btrim(detector_type), '') || '|' ||
                    coalesce(btrim(supplier_id), '')   || '|' ||
                    CASE
                      WHEN coalesce(btrim(item_id), '') <> ''     THEN btrim(item_id)
                      WHEN coalesce(btrim(deal_id), '') <> ''     THEN 'deal:' || btrim(deal_id)
                      WHEN coalesce(btrim(category_id), '') <> '' THEN 'cat:'  || btrim(category_id)
                      ELSE ''
                    END)
           END AS k,
           stage, detected_on, updated_at
      FROM proc.bp_opportunity
     WHERE subject_key IS NULL
), ranked AS (
    SELECT opportunity_id, k,
           row_number() OVER (
               PARTITION BY k
               ORDER BY (stage <> 'identified') DESC,
                        detected_on DESC NULLS LAST,
                        updated_at  DESC NULLS LAST
           ) AS rn
      FROM keyed
     WHERE k IS NOT NULL
)
UPDATE proc.bp_opportunity o
   SET subject_key = r.k
  FROM ranked r
 WHERE o.opportunity_id = r.opportunity_id
   AND r.rn = 1
   AND NOT EXISTS (SELECT 1 FROM proc.bp_opportunity x WHERE x.subject_key = r.k);

CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_opportunity_subject
    ON proc.bp_opportunity (subject_key)
    WHERE subject_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_review_requested
    ON proc.bp_opportunity (review_requested_at)
    WHERE review_requested_at IS NOT NULL;

INSERT INTO proc.bp_lifecycle_transition (object_type, from_state, to_state, note) VALUES
    ('opportunity', 'rejected', 'identified', 're-opened by a person after new information')
ON CONFLICT (object_type, from_state, to_state) DO NOTHING;

COMMIT;
