-- Undoes step 1 of the QDA B1 split. Safe only while the seven old columns still exist on
-- proc.bp_sales_quote / proc.bp_sales_quote_line AND the server is back on code that reads
-- them: a quote drafted on the new code has its cost and margin ONLY in these tables, so
-- copy them back first (below) or they are lost.
BEGIN;

UPDATE proc.bp_sales_quote q
   SET total_cost = i.total_cost, total_margin = i.total_margin, margin_pct = i.margin_pct
  FROM proc.bp_sales_quote_internal i
 WHERE i.sales_quote_id = q.sales_quote_id;

UPDATE proc.bp_sales_quote_line l
   SET unit_cost = i.unit_cost, cost_tier_applied = i.cost_tier_applied,
       line_margin = i.line_margin, line_margin_pct = i.line_margin_pct
  FROM proc.bp_sales_quote_line_internal i
 WHERE i.sales_quote_line_id = l.sales_quote_line_id;

DROP TABLE IF EXISTS proc.bp_sales_quote_line_internal;
DROP TABLE IF EXISTS proc.bp_sales_quote_internal;

COMMIT;
