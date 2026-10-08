-- Restores bp_deal_overview as it stood before 2026-10-08_deal_overview_latest_quote_total.sql
-- (identical in bp_testdb and bp_sqldb): quote_total sums every quote round.

CREATE OR REPLACE VIEW proc.bp_deal_overview AS
 WITH d AS (
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
            sum(d.amount) FILTER (WHERE d.doc_type = 'quote'::text) AS quote_total,
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
