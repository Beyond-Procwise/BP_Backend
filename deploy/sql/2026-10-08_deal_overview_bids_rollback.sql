-- Restores bp_deal_overview as it stood before 2026-10-08_deal_overview_bids.sql (identical in both DBs).
-- CREATE OR REPLACE cannot drop the appended columns, so the view is dropped and recreated;
-- bp_deal_kpis depends on it and is recreated too.

BEGIN;
DROP VIEW proc.bp_deal_kpis;
DROP VIEW proc.bp_deal_overview;
CREATE VIEW proc.bp_deal_overview AS
 WITH d0 AS (
         SELECT bp_deal_documents.deal_id,
            bp_deal_documents.deal_name,
            bp_deal_documents.document_id,
            bp_deal_documents.doc_type,
            bp_deal_documents.doc_pk,
            bp_deal_documents.doc_number,
            bp_deal_documents.doc_date,
            bp_deal_documents.deal_date,
            bp_deal_documents.supplier_id,
            bp_deal_documents.supplier_name,
            bp_deal_documents.buyer_id,
            bp_deal_documents.currency,
            bp_deal_documents.amount,
            bp_deal_documents.amount_incl_tax,
            bp_deal_documents.converted_amount_usd,
            bp_deal_documents.country,
            bp_deal_documents.region,
            bp_deal_documents.confidence_score,
            bp_deal_documents.status,
            bp_deal_documents.created_date
           FROM proc.bp_deal_documents
          WHERE bp_deal_documents.deal_id IS NOT NULL AND bp_deal_documents.deal_id <> ''::text
        ), d AS (
         SELECT d0.deal_id,
            d0.deal_name,
            d0.document_id,
            d0.doc_type,
            d0.doc_pk,
            d0.doc_number,
            d0.doc_date,
            d0.deal_date,
            d0.supplier_id,
            d0.supplier_name,
            d0.buyer_id,
            d0.currency,
            d0.amount,
            d0.amount_incl_tax,
            d0.converted_amount_usd,
            d0.country,
            d0.region,
            d0.confidence_score,
            d0.status,
            d0.created_date,
            d0.doc_type = 'quote'::text AND COALESCE((regexp_match(d0.doc_pk, '\s*\(\s*v(\d+).*\)\s*$'::text, 'i'::text))[1]::integer, 1) < max(COALESCE((regexp_match(d0.doc_pk, '\s*\(\s*v(\d+).*\)\s*$'::text, 'i'::text))[1]::integer, 1)) OVER (PARTITION BY d0.deal_id, d0.doc_type, (regexp_replace(d0.doc_pk, '\s*\(\s*v(\d+).*\)\s*$'::text, ''::text, 'i'::text))) AS superseded_quote
           FROM d0
        ), agg AS (
         SELECT d.deal_id,
            max(d.deal_name::text) AS deal_name,
            max(d.supplier_id) AS supplier_id,
            max(d.supplier_name) AS supplier_name,
            max(d.buyer_id) AS buyer_id,
            max(d.deal_date) AS deal_date,
            min(d.doc_date) AS first_activity_date,
            max(d.doc_date) AS last_activity_date,
            count(*) FILTER (WHERE d.doc_type = 'quote'::text) AS quote_count,
            count(*) FILTER (WHERE d.doc_type = 'po'::text) AS po_count,
            count(*) FILTER (WHERE d.doc_type = 'invoice'::text) AS invoice_count,
            sum(d.amount) FILTER (WHERE d.doc_type = 'quote'::text AND NOT d.superseded_quote) AS quote_total,
            sum(d.amount) FILTER (WHERE d.doc_type = 'po'::text) AS po_total,
            sum(d.amount) FILTER (WHERE d.doc_type = 'invoice'::text) AS invoice_total,
            max(d.currency::text) AS currency,
            sum(d.converted_amount_usd) AS converted_total_usd,
            max(d.doc_date) FILTER (WHERE d.doc_type = 'po'::text) - min(d.doc_date) FILTER (WHERE d.doc_type = 'quote'::text) AS cycle_days_quote_to_po,
            max(d.doc_date) FILTER (WHERE d.doc_type = 'invoice'::text) - min(d.doc_date) FILTER (WHERE d.doc_type = 'po'::text) AS cycle_days_po_to_invoice
           FROM d
          GROUP BY d.deal_id
        ), receipted AS (
         SELECT g.deal_id,
            count(*) AS receipt_count
           FROM proc.bp_goods_receipt_trgt g
          WHERE g.deal_id IS NOT NULL AND g.deal_id::text <> ''::text AND (EXISTS ( SELECT 1
                   FROM proc.bp_goods_receipt_line_items_trgt l
                  WHERE l.grn_id = g.grn_id)) AND COALESCE(g.lines_assessed, 0) > 0
          GROUP BY g.deal_id
        ), gaps AS (
         SELECT sides.deal_id,
            sum(sides.n) AS open_gaps
           FROM ( SELECT i.deal_id,
                    count(*) AS n
                   FROM proc.bp_extraction_discrepancy x
                     JOIN proc.bp_invoice_trgt i ON i.invoice_id = x.doc_pk_candidate
                  WHERE x.doc_type = 'invoice'::text AND x.status = 'open'::text AND (x.issue_type = ANY (ARRAY['billed_not_received'::text, 'nothing_received'::text])) AND i.deal_id IS NOT NULL AND i.deal_id::text <> ''::text
                  GROUP BY i.deal_id
                UNION ALL
                 SELECT g.deal_id,
                    count(*) AS n
                   FROM proc.bp_extraction_discrepancy x
                     JOIN proc.bp_goods_receipt_trgt g ON g.grn_id = x.doc_pk_candidate
                  WHERE x.doc_type = 'goods_receipt'::text AND x.status = 'open'::text AND (x.issue_type = ANY (ARRAY['billed_not_received'::text, 'nothing_received'::text])) AND g.deal_id IS NOT NULL AND g.deal_id::text <> ''::text
                  GROUP BY g.deal_id) sides
          GROUP BY sides.deal_id
        )
 SELECT agg.deal_id,
    agg.deal_name,
    agg.supplier_id,
    agg.supplier_name,
    agg.buyer_id,
    agg.deal_date,
    agg.first_activity_date,
    agg.last_activity_date,
    agg.quote_count,
    agg.po_count,
    agg.invoice_count,
    agg.quote_total,
    agg.po_total,
    agg.invoice_total,
    agg.currency,
    agg.converted_total_usd,
        CASE
            WHEN agg.po_total > 0::numeric THEN round(100.0 * abs(COALESCE(agg.invoice_total, 0::numeric) - agg.po_total) / NULLIF(agg.po_total, 0::numeric), 2)
            ELSE NULL::numeric
        END AS price_variance_pct,
    agg.cycle_days_quote_to_po,
    agg.cycle_days_po_to_invoice,
    agg.quote_count > 0 AS has_quote_anchor,
    agg.quote_count = 0 AND (agg.po_count > 0 OR agg.invoice_count > 0) AS orphaned,
    agg.quote_count > 0 AND agg.po_count > 0 AND agg.invoice_count > 0 AND agg.po_total > 0::numeric AND (abs(COALESCE(agg.invoice_total, 0::numeric) - agg.po_total) / NULLIF(agg.po_total, 0::numeric)) <= 0.10 AS value_reconciled,
        CASE
            WHEN receipted.receipt_count IS NULL THEN NULL::boolean
            ELSE COALESCE(gaps.open_gaps, 0::numeric) = 0::numeric
        END AS three_way_matched
   FROM agg
     LEFT JOIN receipted ON receipted.deal_id::text = agg.deal_id
     LEFT JOIN gaps ON gaps.deal_id::text = agg.deal_id;
CREATE VIEW proc.bp_deal_kpis AS
 SELECT count(*) AS deal_count,
    sum(invoice_total) FILTER (WHERE has_quote_anchor) AS invoiced_total,
    round(avg(cycle_days_po_to_invoice) FILTER (WHERE has_quote_anchor), 1) AS avg_cycle_days,
    round(avg(cycle_days_quote_to_po) FILTER (WHERE has_quote_anchor), 1) AS avg_days_to_po,
    round(100.0 * count(*) FILTER (WHERE value_reconciled)::numeric / NULLIF(count(*) FILTER (WHERE has_quote_anchor), 0)::numeric, 0) AS three_way_match_pct,
    round(100.0 * count(*) FILTER (WHERE three_way_matched)::numeric / NULLIF(count(*) FILTER (WHERE three_way_matched IS NOT NULL), 0)::numeric, 0) AS three_way_matched_pct,
    round(avg(price_variance_pct) FILTER (WHERE has_quote_anchor), 1) AS price_variance_pct,
    ( SELECT count(*) AS count
           FROM ( SELECT bp_invoice_trgt.supplier_id,
                    bp_invoice_trgt.invoice_amount,
                    bp_invoice_trgt.invoice_date
                   FROM proc.bp_invoice_trgt
                  GROUP BY bp_invoice_trgt.supplier_id, bp_invoice_trgt.invoice_amount, bp_invoice_trgt.invoice_date
                 HAVING count(*) > 1) dup) AS duplicate_count,
    sum(invoice_total) FILTER (WHERE po_count = 0) AS no_po_spend,
    count(*) FILTER (WHERE has_quote_anchor) AS complete_deal_count,
    count(*) FILTER (WHERE orphaned) AS orphaned_deal_count,
    COALESCE(sum(COALESCE(po_total, 0::numeric) + COALESCE(invoice_total, 0::numeric)) FILTER (WHERE orphaned), 0::numeric) AS orphaned_spend
   FROM proc.bp_deal_overview;
COMMIT;
