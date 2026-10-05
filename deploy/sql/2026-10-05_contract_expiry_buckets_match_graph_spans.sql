-- 2026-10-05  Contract expiry buckets use the product's graph periods.
--
-- The graphs (SpendIQ engine.js CHART_SPANS) offer 1M, 6M, YTD, 1Y, 2Y. Looking
-- FORWARD at renewals, 1M/6M/1Y/2Y carry over as month edges 1, 6, 12, 24.
-- YTD is left out on purpose: it is calendar-anchored and backward-looking
-- (January to today), so it has no forward equivalent that sits in order
-- between 6 and 12 months.
-- Buckets become: EXPIRED, 0-1, 1-6, 6-12, 12-24; beyond 24 months, no alert.
--
-- Idempotent. Run the detector afterwards: contracts re-bucket, old alerts clear.
UPDATE proc.bp_rule
   SET conditions = jsonb_set(conditions, '{bucket_months}', '[1,6,12,24]'::jsonb),
       version = version + 1,
       last_modified_date = now(),
       last_modified_by = 'graph_spans_2026_10_05'
 WHERE detector_slug = 'contract_expiry_bucket_check'
   AND conditions->'bucket_months' IS DISTINCT FROM '[1,6,12,24]'::jsonb;
