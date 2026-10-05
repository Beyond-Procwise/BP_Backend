-- 2026-10-04  Drop the misnamed column, now that every reader has moved.
--
-- `three_way_match` on proc.bp_deal_overview was a VALUE reconciliation. It is
-- `value_reconciled` since 2026-10-04_deal_overview_three_way.sql, and
-- `three_way_matched` is the real delivery check. Leaving both names alive lets
-- a new reader pick the wrong one, which is how the original misnaming survived
-- for months -- so it goes.
--
-- APPLY THE READERS FIRST. In order:
--   1. BP_Backend  commit "move every reader off three_way_match"
--      (analysis_findings.py, board_paper.py, exec_procurement_summary.py)
--   2. the Node gateway's spendiq.service.ts (o.three_way_match -> o.value_reconciled,
--      three places), DEPLOYED -- it queries this view directly and will 500 on
--      every SpendIQ deal request the moment the column is gone
--   3. then this migration
--
-- CREATE OR REPLACE VIEW cannot drop a column, so both views are rebuilt.
-- proc.bp_deal_kpis depends on bp_deal_overview and is recreated here, in the
-- same transaction, so there is no window where it is missing.
--
-- bp_deal_kpis.three_way_match_pct KEEPS ITS NAME and is computed from
-- value_reconciled -- byte-identical arithmetic, zero behaviour change.
-- Renaming it would be a third cross-repo rename (gateway -> UI -> tiles) for a
-- cosmetic gain, and it is not what §9 of the design asks for. It gains a
-- sibling, three_way_matched_pct, which is the delivery rate over the deals
-- that could actually be assessed -- NULL while the corpus holds no receipts,
-- rather than 0%, because nothing has been checked.

BEGIN;

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
),
-- Which deals have any evidence of delivery that can actually be COMPARED.
-- A receipt whose lines were never read is not evidence: the match skips it
-- (nothing to compare), raises no finding, and counting its header here would
-- turn that silence into "goods received". Found on the live run of
-- 2026-10-05, where the first delivery note reached _trgt with a header, a PO
-- link and zero lines. A deal absent from here has not failed the match; it
-- has not been checked.
receipted AS (
  SELECT g.deal_id, count(*) AS receipt_count
    FROM proc.bp_goods_receipt_trgt g
   WHERE g.deal_id IS NOT NULL AND g.deal_id <> ''
     AND EXISTS (SELECT 1 FROM proc.bp_goods_receipt_line_items_trgt l
                  WHERE l.grn_id = g.grn_id)
   GROUP BY g.deal_id
),
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
  CASE WHEN po_total > 0
       THEN round(100.0 * abs(coalesce(invoice_total, 0) - po_total)
                  / nullif(po_total, 0), 2)
       END AS price_variance_pct,
  cycle_days_quote_to_po, cycle_days_po_to_invoice,
  (quote_count > 0) AS has_quote_anchor,
  (quote_count = 0 AND (po_count > 0 OR invoice_count > 0)) AS orphaned,
  -- do the AMOUNTS agree across quote, PO and invoice?
  (quote_count > 0 AND po_count > 0 AND invoice_count > 0
   AND po_total > 0
   AND abs(coalesce(invoice_total, 0) - po_total) / nullif(po_total, 0) <= 0.10
  ) AS value_reconciled,
  -- did what was billed actually ARRIVE? NULL = not assessed.
  CASE WHEN receipted.receipt_count IS NULL THEN NULL
       ELSE coalesce(gaps.open_gaps, 0) = 0
  END AS three_way_matched
FROM agg
LEFT JOIN receipted ON receipted.deal_id = agg.deal_id
LEFT JOIN gaps      ON gaps.deal_id      = agg.deal_id;

CREATE VIEW proc.bp_deal_kpis AS
SELECT count(*) AS deal_count,
    sum(invoice_total) FILTER (WHERE has_quote_anchor) AS invoiced_total,
    round(avg(cycle_days_po_to_invoice) FILTER (WHERE has_quote_anchor), 1) AS avg_cycle_days,
    round(avg(cycle_days_quote_to_po) FILTER (WHERE has_quote_anchor), 1) AS avg_days_to_po,
    -- Same arithmetic as before, over the renamed column. The KPI keeps its
    -- name so the gateway and the UI tiles do not move for a rename.
    round(100.0 * count(*) FILTER (WHERE value_reconciled)::numeric
          / NULLIF(count(*) FILTER (WHERE has_quote_anchor), 0)::numeric, 0)
      AS three_way_match_pct,
    -- The delivery rate, over the deals that COULD be assessed. NULL while no
    -- deal has a receipt: a 0% would read as "everything failed".
    round(100.0 * count(*) FILTER (WHERE three_way_matched)::numeric
          / NULLIF(count(*) FILTER (WHERE three_way_matched IS NOT NULL), 0)::numeric, 0)
      AS three_way_matched_pct,
    round(avg(price_variance_pct) FILTER (WHERE has_quote_anchor), 1) AS price_variance_pct,
    ( SELECT count(*) AS count
           FROM ( SELECT bp_invoice_trgt.supplier_id,
                    bp_invoice_trgt.invoice_amount,
                    bp_invoice_trgt.invoice_date
                   FROM proc.bp_invoice_trgt
                  GROUP BY bp_invoice_trgt.supplier_id, bp_invoice_trgt.invoice_amount,
                           bp_invoice_trgt.invoice_date
                 HAVING count(*) > 1) dup) AS duplicate_count,
    sum(invoice_total) FILTER (WHERE po_count = 0) AS no_po_spend,
    count(*) FILTER (WHERE has_quote_anchor) AS complete_deal_count,
    count(*) FILTER (WHERE orphaned) AS orphaned_deal_count,
    COALESCE(sum(COALESCE(po_total, 0::numeric) + COALESCE(invoice_total, 0::numeric))
             FILTER (WHERE orphaned), 0::numeric) AS orphaned_spend
   FROM proc.bp_deal_overview;

COMMIT;
