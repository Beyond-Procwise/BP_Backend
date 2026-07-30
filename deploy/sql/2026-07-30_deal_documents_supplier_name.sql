-- bp_deal_documents: resolve supplier_name for quote/invoice rows.
--
-- The 2026-06-11 view hard-coded NULL::text AS supplier_name for the quote and
-- invoice branches (only bp_purchase_order_trgt carries a supplier_name column),
-- so a 15-document deal showed a supplier on exactly its PO rows — 1 of 15 in
-- the Test Data_300726 audit — even though supplier_id was correctly resolved on
-- every row. The join to proc.bp_supplier is what those branches always needed.
-- Column list/order is unchanged, so CREATE OR REPLACE is safe and every
-- dependent view (bp_deal_overview's max(supplier_name)) simply starts seeing
-- real values.

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
    q.created_date
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
    p.created_date
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
    i.created_date
   FROM proc.bp_invoice_trgt i
   LEFT JOIN proc.bp_supplier s ON s.supplier_id = i.supplier_id;
