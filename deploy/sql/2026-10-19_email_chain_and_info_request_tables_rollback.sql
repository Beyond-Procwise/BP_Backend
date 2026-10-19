-- Rolls back 2026-10-19_email_chain_and_info_request_tables.sql. DESTRUCTIVE: drops the rows too.
BEGIN;
DROP TABLE IF EXISTS proc.bp_supplier_info_request;
DROP TABLE IF EXISTS proc.email_thread_map;
DROP TABLE IF EXISTS proc.email_dispatch_chains;
COMMIT;
