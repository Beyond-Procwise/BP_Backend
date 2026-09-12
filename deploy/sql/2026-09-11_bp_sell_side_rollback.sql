-- Rollback of 2026-09-11_bp_sell_side.sql. Children first.
BEGIN;
DROP TABLE IF EXISTS proc.bp_sales_quote_outcome;
DROP TABLE IF EXISTS proc.bp_sales_quote_line;
DROP TABLE IF EXISTS proc.bp_sales_quote;
DROP TABLE IF EXISTS proc.bp_sales_justification;
DROP TABLE IF EXISTS proc.bp_sales_opportunity;
DROP TABLE IF EXISTS proc.bp_account_history_scope;
DROP TABLE IF EXISTS proc.bp_account_contact;
DROP TABLE IF EXISTS proc.bp_account;
COMMIT;
