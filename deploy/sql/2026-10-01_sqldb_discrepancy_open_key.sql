-- Give bp_sqldb the open-findings key that bp_testdb has had since 2026-07-30.
--
-- WHY THIS IS URGENT. `write_discrepancies` (src/services/extraction/persistence.py:400)
-- upserts with
--     ON CONFLICT (doc_type, coalesce(doc_pk_candidate,''), issue_type, coalesce(field_name,''))
--     WHERE coalesce(status,'open') <> 'resolved'
-- Postgres matches that clause to a partial unique index BY SHAPE. bp_sqldb has no such
-- index, so the statement is not merely inefficient -- it is REJECTED outright:
--     ERROR: there is no unique or exclusion constraint matching the ON CONFLICT specification
-- Proved 2026-10-01 by running the same statement against both databases: bp_testdb got
-- past the clause (and failed on an unrelated NOT NULL), bp_sqldb raised the above.
--
-- The consequence, measured: bp_sqldb's newest finding is 2026-07-28, two days before the
-- ON CONFLICT clause shipped. bp_testdb is current (2026-10-01, 5,373 rows). So every
-- extraction finding written against bp_sqldb for the last two months has been LOST, and
-- the failure looks like "this corpus is clean" rather than like an error.
--
-- WHY THIS IS NOT THE JULY MIGRATION REPLAYED. 2026-07-30_discrepancy_dedup.sql DELETED
-- the losing duplicates and kept the FIRST-raised row. Two deliberate differences here:
--
--   1. NOTHING IS DELETED. bp_sqldb holds 152 duplicate key groups / 587 losing rows, and
--      27 of those groups carry materially DIFFERENT figures -- not re-runs of one finding.
--      Example, invoice 01-2024-002, sum_mismatch / invoice_total_incl_tax: seven rows from
--      seven different raw_ids claiming 15,877.50 / 17,065.50 / 17,465.50 / 27,570.50
--      against three different expected totals. That is evidence of either document
--      mis-grouping (seven documents resolved onto one invoice id) or unstable extraction,
--      and deleting six sevenths of it destroys the evidence. They are RESOLVED instead,
--      which takes them out of the index predicate while leaving every row readable, and
--      makes this migration reversible -- the rollback finds them by resolved_by.
--
--   2. THE SURVIVOR IS THE LATEST, not the first. `ON CONFLICT DO UPDATE` overwrites the
--      row with the incoming values, so "latest wins" is what the live upsert would have
--      produced had the index existed all along. Keeping the first would freeze the OLDEST
--      figure, which is the one answer we know the running code would never have left there.
--      This DIVERGES from the July migration on bp_testdb; it is a deliberate choice, noted
--      here so nobody reads the two databases as having been treated identically.
--
-- SIDE EFFECT, quantified before writing: 5 of the 587 losing rows have
-- blocks_promotion = true, across 5 raw_ids. Resolving those fires
-- fn_extraction_discrepancy_resolved, which may pg_notify
-- 'extraction_raw_ready_for_promotion' for a raw_id whose last blocking finding just
-- closed. Nothing was LISTENing on bp_sqldb when this was written (the app runs against
-- bp_testdb per .env), and a missed NOTIFY is recoverable -- promotion.py also sweeps
-- 'pending' rows. If a listener IS attached when this runs, expect up to 5 promotion
-- attempts for July documents.
--
-- Apply with: psql -v ON_ERROR_STOP=1 -f <this file>

BEGIN;

-- 1. Set aside the superseded duplicates, newest row per key surviving as the open one.
--    resolution_action = 'dismiss' is correct and narrow: the raw extracted value was
--    neither applied nor deliberately kept null (promotion.py switches on those two), it
--    is simply not this row's job any more.
--    A HUMAN'S DECISION OUTRANKS RECENCY. status='ignored' means a person looked at the
--    finding and accepted it as risk. The lifecycle table has no ignored -> resolved edge,
--    so bp_lifecycle_guard refuses that move outright -- and it is right to: this migration
--    has no business overwriting a human's call to satisfy an index. An ignored row
--    therefore SURVIVES its key group, and the later open row is the one set aside.
--    Found by running this migration: PO 189990/20 sum_mismatch has 3024 (ignored, by a
--    person) and 3026 (open, a later re-extraction of the same thing). Keeping 3026 on
--    recency alone would have discarded the decision. No key group holds two ignored rows,
--    so this rule is always satisfiable; step 1b proves that rather than trusting it.
WITH ranked AS (
    SELECT discrepancy_id,
           row_number() OVER (
               PARTITION BY doc_type, coalesce(doc_pk_candidate, ''),
                            issue_type, coalesce(field_name, '')
               ORDER BY (status = 'ignored') DESC, discrepancy_id DESC) AS rn
      FROM proc.bp_extraction_discrepancy
     WHERE coalesce(status, 'open') <> 'resolved'
)
UPDATE proc.bp_extraction_discrepancy d
   SET status            = 'resolved',
       resolved_at       = now(),
       resolution_action = 'dismiss',
       resolved_by       = 'sqldb-open-key-migration',
       notes             = coalesce(d.notes, '')
                           || ' [superseded: a later finding holds this key; set aside, not'
                           || ' deleted, so the differing figures stay inspectable]'
  FROM ranked r
 WHERE r.discrepancy_id = d.discrepancy_id
   AND r.rn > 1;

--    Nothing protects the ignored rows here, and nothing needs to: bp_lifecycle_guard
--    REFUSES ignored -> resolved at the row level, so a ranking that tried it aborts this
--    transaction. That is not a theory -- it is how the bug above was found.

-- 1b. Step 2 can only succeed if every key now holds at most one row inside the index
--     predicate. Assert it, because CREATE UNIQUE INDEX's own error names one duplicate
--     tuple and not how many are left, and a migration that fails should say what is wrong.
DO $$
DECLARE remaining INT;
BEGIN
    SELECT count(*) INTO remaining FROM (
        SELECT 1 FROM proc.bp_extraction_discrepancy
         WHERE coalesce(status, 'open') <> 'resolved'
         GROUP BY doc_type, coalesce(doc_pk_candidate, ''),
                  issue_type, coalesce(field_name, '')
        HAVING count(*) > 1) x;
    IF remaining > 0 THEN
        RAISE EXCEPTION
            '% key(s) still hold more than one unresolved finding; the index cannot be built',
            remaining;
    END IF;
END $$;

-- 2. The key itself, character-for-character the definition bp_testdb carries.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_extraction_discrepancy_open_key
    ON proc.bp_extraction_discrepancy
       (doc_type, coalesce(doc_pk_candidate, ''),
        issue_type, coalesce(field_name, ''))
 WHERE coalesce(status, 'open') <> 'resolved';

-- 3. Refuse to finish if the upsert the whole migration exists for still cannot run.
--    A migration that reports success while write_discrepancies stays broken is the
--    failure mode this file is fixing, so it checks rather than assumes.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = 'proc'
           AND indexname = 'ix_bp_extraction_discrepancy_open_key') THEN
        RAISE EXCEPTION 'the open-findings key was not created; write_discrepancies stays broken';
    END IF;
END $$;

COMMIT;
