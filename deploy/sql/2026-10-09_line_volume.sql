-- The volume a line is for, apart from what it bills.
--
-- "Service Desk (24x7, 2,400 users) — annual" bills quantity 1, unit year. The 2,400 users is
-- what is being bought, and comparing bids line by line needs it as a figure: Fortis's £540,000
-- for 2,400 users is £225 a user, dearer than Synapse's £584,000 for 3,000 (£194.67).
-- src/services/extraction/line_volume.py reads it from the line's description into these two
-- columns; quantity and unit_of_measure stay what the document printed. Nullable and additive:
-- nothing existing is changed, and every writer that names its columns is unaffected (the raw
-- writer and the raw->stg->trgt promotions copy the columns the two tables share).
--
-- Apply to BOTH databases (bp_testdb and bp_sqldb) BEFORE deploying the code that writes them:
-- the raw writer inserts every key of a line, so a line carrying `volume` into a table without
-- the column would fail its insert. Then scripts/backfill_line_volume.py fills existing lines.

BEGIN;
ALTER TABLE proc.bp_quote_line_items_raw    ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_quote_line_items_stg    ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_quote_line_items_trgt   ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_po_line_items_raw       ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_po_line_items_stg       ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_po_line_items_trgt      ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_invoice_line_items_raw  ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_invoice_line_items_stg  ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
ALTER TABLE proc.bp_invoice_line_items_trgt ADD COLUMN IF NOT EXISTS volume numeric, ADD COLUMN IF NOT EXISTS volume_unit text;
COMMIT;
