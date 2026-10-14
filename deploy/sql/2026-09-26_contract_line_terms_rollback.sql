-- Rollback for 2026-09-26_contract_line_terms.sql.
-- Remove the contract.yaml line_items block FIRST: load_all_schemas() refuses to start when
-- a declared db_column is missing, so dropping the tables under a live schema breaks start-up.
BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_contracts_contract_id;
DROP TABLE IF EXISTS proc.bp_contract_line_items;
DROP TABLE IF EXISTS proc.bp_contract_line_items_raw;
COMMIT;
