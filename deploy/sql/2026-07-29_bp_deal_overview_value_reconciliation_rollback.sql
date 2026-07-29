-- Rollback for 2026-07-29_bp_deal_overview_value_reconciliation.sql
--
-- Restores the previous presence-only three_way_match definition (matches
-- deploy/sql/2026-06-15_quote_anchor_views.sql). No data loss — this only
-- redefines a view.
BEGIN;

CREATE OR REPLACE VIEW proc.bp_deal_overview AS
WITH d AS (SELECT * FROM proc.bp_deal_documents WHERE deal_id IS NOT NULL AND deal_id <> '')
SELECT
  deal_id,
  max(deal_name) AS deal_name,
  max(supplier_id) AS supplier_id,
  max(supplier_name) AS supplier_name,
  max(buyer_id) AS buyer_id,
  max(deal_date) AS deal_date,
  min(doc_date) AS first_activity_date,
  max(doc_date) AS last_activity_date,
  count(*) FILTER (WHERE doc_type='quote')   AS quote_count,
  count(*) FILTER (WHERE doc_type='po')      AS po_count,
  count(*) FILTER (WHERE doc_type='invoice') AS invoice_count,
  sum(amount) FILTER (WHERE doc_type='quote')   AS quote_total,
  sum(amount) FILTER (WHERE doc_type='po')      AS po_total,
  sum(amount) FILTER (WHERE doc_type='invoice') AS invoice_total,
  max(currency) AS currency,
  sum(converted_amount_usd) AS converted_total_usd,
  (count(*) FILTER (WHERE doc_type='quote')>0
   AND count(*) FILTER (WHERE doc_type='po')>0
   AND count(*) FILTER (WHERE doc_type='invoice')>0) AS three_way_match,
  CASE WHEN sum(amount) FILTER (WHERE doc_type='po') > 0
       THEN round(100.0*abs(coalesce(sum(amount) FILTER (WHERE doc_type='invoice'),0)
              - sum(amount) FILTER (WHERE doc_type='po'))
              / nullif(sum(amount) FILTER (WHERE doc_type='po'),0), 2)
       END AS price_variance_pct,
  (max(doc_date) FILTER (WHERE doc_type='po')
     - min(doc_date) FILTER (WHERE doc_type='quote')) AS cycle_days_quote_to_po,
  (max(doc_date) FILTER (WHERE doc_type='invoice')
     - min(doc_date) FILTER (WHERE doc_type='po')) AS cycle_days_po_to_invoice,
  (count(*) FILTER (WHERE doc_type='quote') > 0) AS has_quote_anchor,
  (count(*) FILTER (WHERE doc_type='quote') = 0
   AND (count(*) FILTER (WHERE doc_type='po') > 0
        OR count(*) FILTER (WHERE doc_type='invoice') > 0)) AS orphaned
FROM d GROUP BY deal_id;

COMMIT;
