BEGIN;
-- Reverses 2026-10-19_bp_gateway_audit.sql. The audit rows are dropped with the tables: export them first if they matter.
DROP TRIGGER IF EXISTS tr_bp_gateway_write_no_truncate ON proc.bp_gateway_write;
DROP TRIGGER IF EXISTS tr_bp_gateway_write_append_only ON proc.bp_gateway_write;
DROP TRIGGER IF EXISTS tr_bp_gateway_call_no_truncate ON proc.bp_gateway_call;
DROP TRIGGER IF EXISTS tr_bp_gateway_call_append_only ON proc.bp_gateway_call;
DROP TABLE IF EXISTS proc.bp_gateway_write;
DROP TABLE IF EXISTS proc.bp_gateway_call;
DROP FUNCTION IF EXISTS proc.bp_gateway_audit_append_only();
COMMIT;
