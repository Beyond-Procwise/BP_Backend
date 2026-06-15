-- 2026-06-15 Quote-anchored deal model: orphan-aware views.
-- bp_deal_overview gains has_quote_anchor + orphaned (appended); bp_deal_kpis
-- computes headline metrics over COMPLETE (quote-anchored) deals and adds
-- complete/orphaned counts + orphaned_spend; new bp_deal_orphans lists the
-- PO/invoice docs awaiting a quote. CREATE OR REPLACE only appends columns.

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
  -- quote-anchored model: a deal is complete only when it contains a quote
  (count(*) FILTER (WHERE doc_type='quote') > 0) AS has_quote_anchor,
  (count(*) FILTER (WHERE doc_type='quote') = 0
   AND (count(*) FILTER (WHERE doc_type='po') > 0
        OR count(*) FILTER (WHERE doc_type='invoice') > 0)) AS orphaned
FROM d GROUP BY deal_id;

-- executive feed: headline metrics over COMPLETE (quote-anchored) deals,
-- plus complete/orphaned counts and orphaned spend.
CREATE OR REPLACE VIEW proc.bp_deal_kpis AS
SELECT
  count(*) AS deal_count,
  sum(invoice_total) FILTER (WHERE has_quote_anchor) AS invoiced_total,
  round(avg(cycle_days_po_to_invoice) FILTER (WHERE has_quote_anchor)::numeric, 1) AS avg_cycle_days,
  round(avg(cycle_days_quote_to_po) FILTER (WHERE has_quote_anchor)::numeric, 1)   AS avg_days_to_po,
  round(100.0*count(*) FILTER (WHERE three_way_match)
        / nullif(count(*) FILTER (WHERE has_quote_anchor),0), 0) AS three_way_match_pct,
  round(avg(price_variance_pct) FILTER (WHERE has_quote_anchor)::numeric, 1) AS price_variance_pct,
  (SELECT count(*) FROM (
      SELECT supplier_id, invoice_amount, invoice_date FROM proc.bp_invoice_trgt
      GROUP BY supplier_id, invoice_amount, invoice_date HAVING count(*)>1) dup) AS duplicate_count,
  sum(invoice_total) FILTER (WHERE po_count=0) AS no_po_spend,
  count(*) FILTER (WHERE has_quote_anchor)  AS complete_deal_count,
  count(*) FILTER (WHERE orphaned)          AS orphaned_deal_count,
  coalesce(sum(coalesce(po_total,0)+coalesce(invoice_total,0)) FILTER (WHERE orphaned),0) AS orphaned_spend
FROM proc.bp_deal_overview;

-- orphaned PO/invoice documents awaiting their anchoring quote
CREATE OR REPLACE VIEW proc.bp_deal_orphans AS
SELECT dd.deal_id, dd.deal_name, dd.doc_type, dd.doc_pk, dd.doc_number,
       dd.supplier_id, dd.supplier_name, dd.amount, dd.currency, dd.doc_date,
       dd.status
FROM proc.bp_deal_documents dd
WHERE dd.doc_type IN ('po','invoice')
  AND NOT EXISTS (
      SELECT 1 FROM proc.bp_quote_trgt q
      WHERE q.deal_id = dd.deal_id AND dd.deal_id IS NOT NULL AND dd.deal_id <> '');
