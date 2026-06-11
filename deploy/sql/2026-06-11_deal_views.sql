-- 2026-06-11 Deal-centric read views. CREATE OR REPLACE, read-only.
-- 7a. one row per document in a deal
CREATE OR REPLACE VIEW proc.bp_deal_documents AS
SELECT deal_id, deal_name, document_id, 'quote'::text AS doc_type,
       quote_id AS doc_pk, quote_id AS doc_number, quote_date AS doc_date, deal_date,
       supplier_id, NULL::text AS supplier_name, buyer_id, currency,
       total_amount AS amount, total_amount_incl_tax AS amount_incl_tax,
       converted_amount_usd, country, region, confidence_score,
       status, created_date
FROM proc.bp_quote_trgt
UNION ALL
SELECT deal_id, deal_name, document_id, 'po', po_id, po_id, order_date, deal_date,
       supplier_id, supplier_name, buyer_id, currency,
       total_amount, total_amount_incl_tax, converted_amount_usd,
       ship_to_country, delivery_region, confidence_score, po_status, created_date
FROM proc.bp_purchase_order_trgt
UNION ALL
SELECT deal_id, deal_name, document_id, 'invoice', invoice_id, invoice_id, invoice_date, deal_date,
       supplier_id, NULL, buyer_id, currency,
       invoice_amount, invoice_total_incl_tax, converted_amount_usd,
       country, region, confidence_score, invoice_status, created_date
FROM proc.bp_invoice_trgt;

-- 7b. one row per deal
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
     - min(doc_date) FILTER (WHERE doc_type='po')) AS cycle_days_po_to_invoice
FROM d GROUP BY deal_id;

-- 7c. executive dashboard single-row feed
CREATE OR REPLACE VIEW proc.bp_deal_kpis AS
SELECT
  count(*) AS deal_count,
  sum(invoice_total) AS invoiced_total,
  round(avg(cycle_days_po_to_invoice)::numeric, 1) AS avg_cycle_days,
  round(avg(cycle_days_quote_to_po)::numeric, 1)   AS avg_days_to_po,
  round(100.0*count(*) FILTER (WHERE three_way_match)/nullif(count(*),0),0) AS three_way_match_pct,
  round(avg(price_variance_pct)::numeric, 1) AS price_variance_pct,
  (SELECT count(*) FROM (
      SELECT supplier_id, invoice_amount, invoice_date FROM proc.bp_invoice_trgt
      GROUP BY supplier_id, invoice_amount, invoice_date HAVING count(*)>1) dup) AS duplicate_count,
  sum(invoice_total) FILTER (WHERE po_count=0) AS no_po_spend
FROM proc.bp_deal_overview;

-- 7d. document processing status board
CREATE OR REPLACE VIEW proc.bp_process_monitor_status AS
SELECT id, file_path, document_type AS doc_type, category, deal_id, deal_name,
       status, start_ts, end_ts, lastmodified_date
FROM proc.process_monitor;
