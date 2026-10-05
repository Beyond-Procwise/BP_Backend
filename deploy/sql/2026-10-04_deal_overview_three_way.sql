-- 2026-10-04  The deal overview separates "the amounts reconcile" from
--             "the goods arrived".
--
-- `three_way_match` has never been a three-way match. It checks that a quote, a
-- PO and an invoice all exist and that the invoice total is within 10% of the
-- PO total (2026-07-29_bp_deal_overview_value_reconciliation.sql). That is a
-- real and useful check -- a VALUE reconciliation across three document types --
-- and it is not evidence that anything was delivered. A buyer reading
-- "three-way matched" believes somebody confirmed the goods arrived.
--
-- Two columns, APPENDED (CREATE OR REPLACE VIEW can only append -- it cannot
-- reorder or rename an existing column, which is also why `three_way_match`
-- survives this migration and is dropped separately once every reader has
-- moved):
--
--   value_reconciled   byte-for-byte the expression three_way_match computes
--                      today. Not one deal's answer changes.
--                      tests/sql/test_deal_overview_three_way.py asserts that
--                      directly, row by row, rather than trusting this comment.
--
--   three_way_matched  did what was billed actually arrive? NULL when the deal
--                      has no goods receipt at all, which is the honest answer
--                      and is NOT `false`: missing paperwork must not
--                      manufacture a failure rate (design section 11, Review
--                      Focus #2). Where a receipt exists the verdict is driven
--                      by the findings the match actually raised, so `true`
--                      can never mean "nothing looked".
--
-- Today every deal reads NULL, because the corpus holds zero goods receipts
-- (design section 13). That is the point: the column says "not assessed" out
-- loud instead of a confident answer nobody earned.
--
-- Additive: CREATE OR REPLACE VIEW, same name, two new columns at the end.

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
),
-- Which deals have any evidence of delivery at all. A deal absent from here
-- has not failed the match; it has not been checked.
receipted AS (
  SELECT deal_id, count(*) AS receipt_count
    FROM proc.bp_goods_receipt_trgt
   WHERE deal_id IS NOT NULL AND deal_id <> ''
   GROUP BY deal_id
),
-- Open quantity gaps the match raised, reached from either side: the finding is
-- filed against whichever document was being read when it was raised, so an
-- invoice-side finding joins through bp_invoice_trgt and a receipt-side one
-- through bp_goods_receipt_trgt.
gaps AS (
  SELECT deal_id, sum(n) AS open_gaps FROM (
    SELECT i.deal_id, count(*) AS n
      FROM proc.bp_extraction_discrepancy x
      JOIN proc.bp_invoice_trgt i ON i.invoice_id = x.doc_pk_candidate
     WHERE x.doc_type = 'invoice' AND x.status = 'open'
       AND x.issue_type IN ('billed_not_received','nothing_received')
       AND i.deal_id IS NOT NULL AND i.deal_id <> ''
     GROUP BY i.deal_id
    UNION ALL
    SELECT g.deal_id, count(*) AS n
      FROM proc.bp_extraction_discrepancy x
      JOIN proc.bp_goods_receipt_trgt g ON g.grn_id = x.doc_pk_candidate
     WHERE x.doc_type = 'goods_receipt' AND x.status = 'open'
       AND x.issue_type IN ('billed_not_received','nothing_received')
       AND g.deal_id IS NOT NULL AND g.deal_id <> ''
     GROUP BY g.deal_id
  ) sides GROUP BY deal_id
)
SELECT
  agg.deal_id, deal_name, supplier_id, supplier_name, buyer_id, deal_date,
  first_activity_date, last_activity_date,
  quote_count, po_count, invoice_count,
  quote_total, po_total, invoice_total, currency, converted_total_usd,
  -- KEPT, unchanged, until Task 10 has moved every reader. Dropped by
  -- 2026-10-04_deal_overview_drop_three_way_match.sql.
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
  (quote_count = 0 AND (po_count > 0 OR invoice_count > 0)) AS orphaned,
  -- The same expression as three_way_match above, under the name that says
  -- what it measures.
  (quote_count > 0 AND po_count > 0 AND invoice_count > 0
   AND po_total > 0
   AND abs(coalesce(invoice_total, 0) - po_total) / nullif(po_total, 0) <= 0.10
  ) AS value_reconciled,
  CASE WHEN receipted.receipt_count IS NULL THEN NULL
       ELSE coalesce(gaps.open_gaps, 0) = 0
  END AS three_way_matched
FROM agg
LEFT JOIN receipted ON receipted.deal_id = agg.deal_id
LEFT JOIN gaps      ON gaps.deal_id      = agg.deal_id;
