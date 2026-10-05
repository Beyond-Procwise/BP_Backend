-- Reverses deploy/sql/2026-10-04_deal_overview_three_way.sql by restoring the
-- view definition from 2026-07-29_bp_deal_overview_value_reconciliation.sql.
-- CREATE OR REPLACE cannot DROP a column, so the view must be dropped and
-- rebuilt. proc.bp_deal_kpis DEPENDS on it, so CASCADE is required and
-- bp_deal_kpis must then be recreated from its own migration
-- (deploy/sql/2026-06-11_deal_views.sql, then 2026-06-15_quote_anchor_views.sql). Running
-- this without that second step leaves the KPI view missing, which is a
-- bigger outage than the two columns this undoes -- so this rollback is
-- deliberately NOT a one-liner, and the second step is not optional.
--
--   1. psql -f deploy/sql/2026-10-04_deal_overview_three_way_rollback.sql
--   2. psql -f deploy/sql/2026-06-11_deal_views.sql
--   3. psql -f deploy/sql/2026-06-15_quote_anchor_views.sql
DROP VIEW IF EXISTS proc.bp_deal_overview CASCADE;

CREATE VIEW proc.bp_deal_overview AS
WITH d AS (
  SELECT * FROM proc.bp_deal_documents WHERE deal_id IS NOT NULL AND deal_id <> ''
),
agg AS (
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
    (max(doc_date) FILTER (WHERE doc_type='po')
       - min(doc_date) FILTER (WHERE doc_type='quote')) AS cycle_days_quote_to_po,
    (max(doc_date) FILTER (WHERE doc_type='invoice')
       - min(doc_date) FILTER (WHERE doc_type='po')) AS cycle_days_po_to_invoice
  FROM d GROUP BY deal_id
)
SELECT
  deal_id, deal_name, supplier_id, supplier_name, buyer_id, deal_date,
  first_activity_date, last_activity_date,
  quote_count, po_count, invoice_count,
  quote_total, po_total, invoice_total, currency, converted_total_usd,
  (quote_count > 0 AND po_count > 0 AND invoice_count > 0
   AND po_total > 0
   AND abs(coalesce(invoice_total, 0) - po_total) / nullif(po_total, 0) <= 0.10
  ) AS three_way_match,
  CASE WHEN po_total > 0
       THEN round(100.0 * abs(coalesce(invoice_total, 0) - po_total)
                  / nullif(po_total, 0), 2)
       END AS price_variance_pct,
  cycle_days_quote_to_po, cycle_days_po_to_invoice,
  (quote_count > 0) AS has_quote_anchor,
  (quote_count = 0 AND (po_count > 0 OR invoice_count > 0)) AS orphaned
FROM agg;
