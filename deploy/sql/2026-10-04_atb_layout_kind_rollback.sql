-- Rollback for 2026-10-04_atb_layout_kind.sql. Dropping the column also drops its constraint.
-- Any composed pages already imported become indistinguishable from templates, which is why
-- this is a rollback and not a migration path.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_page_layout_pack_kind;
ALTER TABLE proc.bp_page_layout DROP COLUMN IF EXISTS kind;
COMMIT;
