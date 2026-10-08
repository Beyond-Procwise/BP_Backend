-- Every bid is a document of its deal (2026-10-09, user ruling "every bid on the deal").
--
-- Two routes formed deals differently: a confirmed proposal put every rival bid on
-- bp_deal_documents, while _attach_rival_quotes (PO-anchored deals) marked rivals
-- 'not_awarded' and this view dropped them -- the same sourcing event read as 3 bids on one
-- route and 1 on the other, and the comparison screens could not see the rivals.
--   bp_deal_documents  rivals included; award_status appended as the last column
--   bp_deal_overview   bid_count / lowest_bid / quote_count now see the rivals; the deal's
--                      VALUE (quote_total) and quote-to-PO cycle still ignore any bid marked
--                      not_awarded (as does converted_total_usd), so they are unchanged
--                      wherever no rival is attached.
-- Supersedes the not_awarded filter in the 2026-09-26 quote-sourcing-event migration:
-- re-applying that file would drop the rivals again.
-- Rollback: 2026-10-09_deal_documents_every_bid_rollback.sql

CREATE OR REPLACE VIEW proc.bp_deal_documents AS
 SELECT q.deal_id,
    q.deal_name,
    q.document_id,
    'quote'::text AS doc_type,
    q.quote_id AS doc_pk,
    q.quote_id AS doc_number,
    q.quote_date AS doc_date,
    q.deal_date,
    q.supplier_id,
    s.supplier_name,
    q.buyer_id,
    q.currency,
    q.total_amount AS amount,
    q.total_amount_incl_tax AS amount_incl_tax,
    q.converted_amount_usd,
    q.country,
    q.region,
    q.confidence_score,
    q.status,
    q.created_date,
    q.award_status AS award_status
   FROM proc.bp_quote_trgt q
     LEFT JOIN proc.bp_supplier s ON s.supplier_id = q.supplier_id
UNION ALL
 SELECT p.deal_id,
    p.deal_name,
    p.document_id,
    'po'::text AS doc_type,
    p.po_id AS doc_pk,
    p.po_id AS doc_number,
    p.order_date AS doc_date,
    p.deal_date,
    p.supplier_id,
    COALESCE(p.supplier_name, s.supplier_name) AS supplier_name,
    p.buyer_id,
    p.currency,
    p.total_amount AS amount,
    p.total_amount_incl_tax AS amount_incl_tax,
    p.converted_amount_usd,
    p.ship_to_country AS country,
    p.delivery_region AS region,
    p.confidence_score,
    p.po_status AS status,
    p.created_date,
    NULL::text AS award_status
   FROM proc.bp_purchase_order_trgt p
     LEFT JOIN proc.bp_supplier s ON s.supplier_id = p.supplier_id
UNION ALL
 SELECT i.deal_id,
    i.deal_name,
    i.document_id,
    'invoice'::text AS doc_type,
    i.invoice_id AS doc_pk,
    i.invoice_id AS doc_number,
    i.invoice_date AS doc_date,
    i.deal_date,
    i.supplier_id,
    s.supplier_name,
    i.buyer_id,
    i.currency,
    i.invoice_amount AS amount,
    i.invoice_total_incl_tax AS amount_incl_tax,
    i.converted_amount_usd,
    i.country,
    i.region,
    i.confidence_score,
    i.invoice_status AS status,
    i.created_date,
    NULL::text AS award_status
   FROM proc.bp_invoice_trgt i
     LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id;

CREATE OR REPLACE VIEW proc.bp_deal_overview AS
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
            bp_deal_documents.created_date,
            bp_deal_documents.award_status
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
            d0.award_status,
            d0.doc_type = 'quote'::text AND COALESCE((regexp_match(d0.doc_pk, '\s*\(\s*v(\d+).*\)\s*$'::text, 'i'::text))[1]::integer, 1) < max(COALESCE((regexp_match(d0.doc_pk, '\s*\(\s*v(\d+).*\)\s*$'::text, 'i'::text))[1]::integer, 1)) OVER (PARTITION BY d0.deal_id, d0.doc_type, (regexp_replace(d0.doc_pk, '\s*\(\s*v(\d+).*\)\s*$'::text, ''::text, 'i'::text))) AS superseded_quote,
                CASE
                    WHEN d0.doc_type = 'quote'::text THEN COALESCE((regexp_match(d0.doc_pk, '\(\s*V(\d+)'::text, 'i'::text))[1]::integer, 1)
                    ELSE NULL::integer
                END AS quote_version,
            max(d0.supplier_id) FILTER (WHERE d0.doc_type = ANY (ARRAY['po'::text, 'invoice'::text])) OVER (PARTITION BY d0.deal_id) AS awarded_supplier_id,
            max(d0.supplier_name) FILTER (WHERE d0.doc_type = ANY (ARRAY['po'::text, 'invoice'::text])) OVER (PARTITION BY d0.deal_id) AS awarded_supplier_name
           FROM d0
        ), agg AS (
         SELECT d.deal_id,
            max(d.deal_name::text) AS deal_name,
            COALESCE(max(d.awarded_supplier_id),
                CASE
                    WHEN count(DISTINCT d.supplier_id) FILTER (WHERE d.doc_type = 'quote'::text) = 1 THEN max(d.supplier_id) FILTER (WHERE d.doc_type = 'quote'::text)
                    ELSE NULL::text
                END) AS supplier_id,
            COALESCE(max(d.awarded_supplier_name),
                CASE
                    WHEN count(DISTINCT d.supplier_id) FILTER (WHERE d.doc_type = 'quote'::text) = 1 THEN max(d.supplier_name) FILTER (WHERE d.doc_type = 'quote'::text)
                    ELSE NULL::text
                END) AS supplier_name,
            max(d.buyer_id) AS buyer_id,
            max(d.deal_date) AS deal_date,
            min(d.doc_date) AS first_activity_date,
            max(d.doc_date) AS last_activity_date,
            count(*) FILTER (WHERE d.doc_type = 'quote'::text) AS quote_count,
            count(*) FILTER (WHERE d.doc_type = 'po'::text) AS po_count,
            count(*) FILTER (WHERE d.doc_type = 'invoice'::text) AS invoice_count,
            COALESCE(sum(d.amount) FILTER (WHERE d.doc_type = 'quote'::text AND NOT d.superseded_quote AND d.supplier_id = d.awarded_supplier_id AND d.award_status IS DISTINCT FROM 'not_awarded'::text), min(d.amount) FILTER (WHERE d.doc_type = 'quote'::text AND NOT d.superseded_quote AND d.award_status IS DISTINCT FROM 'not_awarded'::text), min(d.amount) FILTER (WHERE d.doc_type = 'quote'::text AND NOT d.superseded_quote)) AS quote_total,
            count(*) FILTER (WHERE d.doc_type = 'quote'::text AND NOT d.superseded_quote) AS bid_count,
            max(d.quote_version) AS max_quote_version,
            min(d.amount) FILTER (WHERE d.doc_type = 'quote'::text AND NOT d.superseded_quote) AS lowest_bid,
            sum(d.amount) FILTER (WHERE d.doc_type = 'po'::text) AS po_total,
            sum(d.amount) FILTER (WHERE d.doc_type = 'invoice'::text) AS invoice_total,
            max(d.currency::text) AS currency,
            sum(d.converted_amount_usd) FILTER (WHERE NOT d.superseded_quote AND d.award_status IS DISTINCT FROM 'not_awarded'::text) AS converted_total_usd,
            max(d.doc_date) FILTER (WHERE d.doc_type = 'po'::text) - min(d.doc_date) FILTER (WHERE d.doc_type = 'quote'::text AND d.award_status IS DISTINCT FROM 'not_awarded'::text) AS cycle_days_quote_to_po,
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
        END AS three_way_matched,
    agg.bid_count,
    agg.max_quote_version,
    agg.lowest_bid
   FROM agg
     LEFT JOIN receipted ON receipted.deal_id::text = agg.deal_id
     LEFT JOIN gaps ON gaps.deal_id::text = agg.deal_id;
