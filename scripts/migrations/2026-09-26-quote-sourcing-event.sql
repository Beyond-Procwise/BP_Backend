-- Quote traceability: record the sourcing event a quote answers, and whether it won.
--
-- WHY
-- A requirement goes to market and 2-5 suppliers bid. Only the winning bid becomes a
-- purchase order, and a deal is formed by anchoring a quote to that PO (po_quote_anchor,
-- strict 1:1). A losing bid therefore has no PO, no route into a deal, and -- because
-- document_id is minted as '{deal_id}::quote::{quote_id}' -- no document_id either.
--
-- Nothing in the schema recorded that the rival bids answered the same requirement, so
-- the fan-out was lost the moment the rows were written. On the seeded corpus that left
-- 15,998 of 21,054 quotes attached to no deal, and proc.bp_supplier_ranking -- which
-- reads competing quotes per deal -- with 3 rows in total.
--
-- requirement_id is the general key: for real documents it is the RFQ / tender reference
-- the quote cites. award_status records the outcome so that "what did this deal cost"
-- and "how many bids did we get" stay different questions.
--
-- Both columns are nullable. A quote whose sourcing event is unknown keeps NULL rather
-- than being guessed into a group, and a quote whose outcome is unknown keeps NULL
-- rather than being presumed lost.

ALTER TABLE proc.bp_quote_trgt
    ADD COLUMN IF NOT EXISTS requirement_id TEXT,
    ADD COLUMN IF NOT EXISTS award_status   TEXT;

ALTER TABLE proc.bp_quote_trgt
    DROP CONSTRAINT IF EXISTS bp_quote_trgt_award_status_check;
ALTER TABLE proc.bp_quote_trgt
    ADD CONSTRAINT bp_quote_trgt_award_status_check
    CHECK (award_status IS NULL OR award_status IN ('awarded', 'not_awarded'));

-- Competing bids are always fetched as a set, by sourcing event.
CREATE INDEX IF NOT EXISTS ix_bp_quote_trgt_requirement_id
    ON proc.bp_quote_trgt (requirement_id);

-- Supplier ranking reads quotes per deal; this is the access path it uses.
CREATE INDEX IF NOT EXISTS ix_bp_quote_trgt_deal_id
    ON proc.bp_quote_trgt (deal_id);

-- proc.bp_deal_documents is the deal's DOCUMENT CHAIN: the documents that constitute
-- the transaction and that three-way matching runs over. It is a view over the _trgt
-- tables, so giving a losing bid a deal_id would otherwise slide it straight into that
-- chain -- and into proc.bp_deal_overview, where quote_count and quote_total are read
-- off it. A deal that received four bids would report four "quotes" and a quote_total
-- that is the sum of every offer rather than the one that was accepted.
--
-- A rival bid is evidence about the sourcing event, not part of the transaction. It
-- stays reachable through proc.bp_quote_trgt (deal_id, requirement_id), which is what
-- supplier ranking reads, and is excluded here. NULL award_status is kept: a quote
-- whose outcome nobody recorded is not thereby a loser.
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
   FROM (proc.bp_quote_trgt q
     LEFT JOIN proc.bp_supplier s ON ((s.supplier_id = q.supplier_id)))
  WHERE q.award_status IS DISTINCT FROM 'not_awarded'
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
   FROM (proc.bp_purchase_order_trgt p
     LEFT JOIN proc.bp_supplier s ON ((s.supplier_id = p.supplier_id)))
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
   FROM (proc.bp_invoice_trgt i
     LEFT JOIN proc.bp_supplier s ON ((s.supplier_id = i.supplier_id)));
