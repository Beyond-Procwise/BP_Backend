-- Rollback for 2026-08-07_canonical_fdw.sql
--
-- ORDERING WARNING. Drop the views before the foreign tables they read, and
-- the user mapping before the server it belongs to. DROP SCHEMA ... CASCADE
-- handles the first pair, but dropping the server while a mapping still
-- references it fails -- and on a re-run the half-completed rollback then
-- fails at a different statement, which is how a rollback stops being
-- idempotent exactly when it is most likely to be re-run.
--
-- NOT destructive to any data: everything here is a pointer. The canonical
-- rows live in uicanvas and are untouched by this. The extension is left
-- installed, since other work may depend on it.
BEGIN;

DROP VIEW IF EXISTS proc.bp_contract_master;
DROP VIEW IF EXISTS proc.bp_supplier_master;
DROP VIEW IF EXISTS proc.bp_category_product_map;
DROP VIEW IF EXISTS proc.bp_product_master;
DROP VIEW IF EXISTS proc.bp_category_master;

DROP SCHEMA IF EXISTS canonical CASCADE;

DROP USER MAPPING IF EXISTS FOR CURRENT_USER SERVER uicanvas_srv;
DROP SERVER IF EXISTS uicanvas_srv CASCADE;

COMMIT;
