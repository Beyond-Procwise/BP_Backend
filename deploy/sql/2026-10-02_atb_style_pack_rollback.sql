-- Rollback for 2026-10-02_atb_style_pack.sql. Drops both tables and everything in them: an
-- imported pack is re-derivable from its source file, so nothing here is irreplaceable.
BEGIN;
DROP TABLE IF EXISTS proc.bp_page_layout;
DROP TABLE IF EXISTS proc.bp_style_pack;
COMMIT;
