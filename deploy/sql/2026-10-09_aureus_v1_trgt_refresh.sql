-- Aureus AUR-2025-0619 (V1): bring _trgt in line with its corrected _stg.
--
-- Read on 2026-07-30 (raw 38690), the table reader took the order form's "3-yr subtotal (£)"
-- column as each line's cost: licence 3,485,184, support 304,410, sandbox 136,674 (Year 1 is
-- 1,122,000 / 98,000 / 44,000), and the header total became 3,485,184 (the licence line's
-- three-year figure). The document states "Year 1 total (ex-VAT) 1,326,000"; a quote's total is
-- its Year 1 (the user's ruling). The re-read of 2026-10-01 (raw 38848) is right and refreshed
-- _stg, but a re-read never refreshes _trgt, so the deal's analysis still showed Aureus V1 at
-- about three times its price. The raw rows are not touched: they are the record of each read.
--
-- Guarded: a row changes only while it still holds the 2026-07-30 figure, so a re-run, or a
-- _trgt that has since been refreshed some other way, is a no-op. bp_testdb only (bp_sqldb does
-- not hold this quote).
BEGIN;
UPDATE proc.bp_quote_line_items_trgt t
   SET line_total = s.line_total,
       last_modified_by = 'deploy/sql/2026-10-09_aureus_v1_trgt_refresh.sql',
       last_modified_date = now()
  FROM proc.bp_quote_line_items_stg s
 WHERE t.quote_id = 'AUR-2025-0619' AND s.quote_id = t.quote_id AND s.line_number = t.line_number
   AND (t.line_number, t.line_total) IN ((1, 3485184.00), (2, 304410.00), (3, 136674.00))
   AND s.line_total IS NOT NULL AND s.line_total <> t.line_total;

UPDATE proc.bp_quote_trgt
   SET total_amount = 1326000.00,
       last_modified_by = 'deploy/sql/2026-10-09_aureus_v1_trgt_refresh.sql',
       last_modified_date = now()
 WHERE quote_id = 'AUR-2025-0619' AND total_amount = 3485184.00;

SELECT quote_id, line_number, line_total FROM proc.bp_quote_line_items_trgt
 WHERE quote_id = 'AUR-2025-0619' ORDER BY line_number;
SELECT quote_id, total_amount FROM proc.bp_quote_trgt WHERE quote_id = 'AUR-2025-0619';
COMMIT;
