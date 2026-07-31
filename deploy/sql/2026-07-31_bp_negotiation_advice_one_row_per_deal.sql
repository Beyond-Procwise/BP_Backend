-- deploy/sql/2026-07-31_bp_negotiation_advice_one_row_per_deal.sql
-- One advice row per deal, so save_advice can upsert instead of inserting.
--
-- The Negotiate dashboard rebuilds advice on every view, and every build wrote a
-- new row: a deal left open in a browser grew the table a row per refresh. The
-- rows were also the anchor for buyer-stated facts, so each rebuild moved the
-- advice_id out from under the facts already stated against it — a fact stated
-- on one turn was invisible by the next.
--
-- Collapses any existing history to the newest row per deal, moving that deal's
-- facts onto the survivor first. Additive + idempotent; safe to re-run.
BEGIN;

CREATE TEMP TABLE _advice_survivor ON COMMIT DROP AS
SELECT DISTINCT ON (deal_id) deal_id, advice_id
FROM proc.bp_negotiation_advice
ORDER BY deal_id, created_at DESC NULLS LAST, advice_id;

-- Re-point facts from superseded rows onto the survivor, unless the survivor
-- already carries that key (its value is the more recent statement, so it wins).
UPDATE proc.bp_negotiation_advice_fact f
SET advice_id = s.advice_id
FROM proc.bp_negotiation_advice a
JOIN _advice_survivor s ON s.deal_id = a.deal_id
WHERE f.advice_id = a.advice_id
  AND f.advice_id <> s.advice_id
  AND NOT EXISTS (
      SELECT 1 FROM proc.bp_negotiation_advice_fact g
      WHERE g.advice_id = s.advice_id AND g.fact_key = f.fact_key
  );

-- Whatever could not move was superseded by the survivor's own value.
DELETE FROM proc.bp_negotiation_advice_fact f
USING proc.bp_negotiation_advice a
WHERE f.advice_id = a.advice_id
  AND NOT EXISTS (
      SELECT 1 FROM _advice_survivor s WHERE s.advice_id = a.advice_id
  );

DELETE FROM proc.bp_negotiation_advice a
WHERE NOT EXISTS (
    SELECT 1 FROM _advice_survivor s WHERE s.advice_id = a.advice_id
);

-- ON CONFLICT (deal_id) needs a unique index. It also subsumes the plain
-- deal_id index created by 2026-07-29_bp_negotiation_advice.sql, so the name is
-- reused rather than leaving two indexes on the same column.
DROP INDEX IF EXISTS proc.ix_bp_negotiation_advice_deal_id;
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_negotiation_advice_deal_id
    ON proc.bp_negotiation_advice (deal_id);

COMMIT;
