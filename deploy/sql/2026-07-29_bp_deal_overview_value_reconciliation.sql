-- 2026-07-29 Make three_way_match check that quote/PO/invoice amounts
-- actually reconcile, not just that all three document types exist.
--
-- Found live on DEALV2-005049 (Kestrel Supplies): quote_total == po_total ==
-- INR 14.56M, but three separate invoices each billed the full PO amount
-- again (invoice_total = INR 43.69M, price_variance_pct = 200). The deal
-- still showed three_way_match = true, because that column only checked
-- "count(quote)>0 AND count(po)>0 AND count(invoice)>0" — never that the
-- amounts agreed. price_variance_pct was computed correctly all along but
-- nothing (badge or opportunity miner) ever read it.
--
-- Checked live: 3,388 of 5,039 deals have >20% price variance and would flip
-- from true to false under this definition — this was not specific to one
-- deal. A 10% tolerance is used (matches the new Invoice Overbilling
-- detector's default threshold) so "three_way_match" and "no overbilling
-- opportunity raised" describe the same thing.
--
-- Restructured as a two-level CTE (agg then final SELECT) because Postgres
-- SELECT-list aliases aren't visible to sibling expressions in the same
-- SELECT — three_way_match and price_variance_pct both need the aggregated
-- totals, so they're computed from the agg CTE rather than duplicating the
-- FILTER expressions inline twice.
--
-- Additive: same columns, same view name, CREATE OR REPLACE only.

CREATE OR REPLACE VIEW proc.bp_deal_overview AS
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
  -- amount reconciliation, not just document presence: all three doc types
  -- must exist AND the invoice total must be within 10% of the PO total.
  -- (column order must match the prior view definition — CREATE OR REPLACE
  -- VIEW can only append columns, never reorder or rename existing ones.)
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
