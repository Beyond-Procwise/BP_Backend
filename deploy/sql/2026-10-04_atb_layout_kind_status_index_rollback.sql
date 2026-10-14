-- Rollback of 2026-10-04_atb_layout_kind_status_index.sql. Dropping an index loses no data.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_page_layout_status_kind;
COMMIT;
