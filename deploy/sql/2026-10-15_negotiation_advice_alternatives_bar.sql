-- The "many alternatives" bar becomes a supplier count: 3 (lever audit rec. 4).
--
-- It was 93, the median per-deal count when it was seeded (2026-07-29). The
-- count grows with the data -- 407 on bp_testdb by 2026-10-15 -- so every test
-- deal read as a contested market, while on bp_sqldb 14 of 15 deals have only
-- their own supplier on record and every one read as a thin market. Three or
-- more suppliers on record quoting the deal's items is a contested market;
-- one (the deal's own) is no data, and the code reads it as no verdict.
--
-- Changes the row only while it still holds the seeded 93, so a value someone
-- has since set is left alone. Approved by the user 2026-10-15. Apply to both
-- bp_testdb and bp_sqldb.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{many_alternatives}', '3'::jsonb),
       version = COALESCE(version, 1) + 1,
       last_modified_date = now(),
       last_modified_by = 'negotiation_advice'
 WHERE policy_name = 'negotiation_advice_thresholds'
   AND (policy_details->>'many_alternatives')::numeric = 93;

COMMIT;
