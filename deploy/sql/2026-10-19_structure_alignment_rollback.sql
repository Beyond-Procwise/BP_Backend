-- Rolls back 2026-10-19_structure_alignment.sql. DESTRUCTIVE: drops the tables and their rows, and the two columns' values.
BEGIN;
DROP TABLE IF EXISTS proc.bp_agent_group;
DROP TABLE IF EXISTS proc.bp_egress_event;
DROP TABLE IF EXISTS proc.bp_link_rejection;
DROP TABLE IF EXISTS proc.email_dispatch;
DROP TABLE IF EXISTS proc.supplier_risk_scores;
DROP TABLE IF EXISTS proc.supplier_risk_signals;
DROP TABLE IF EXISTS proc.bp_standin_actor_backfill;
DROP TABLE IF EXISTS proc.bp_supplier_id_crosswalk;
ALTER TABLE proc.bp_workflow_run DROP COLUMN IF EXISTS initiated_by;
ALTER TABLE proc.negotiation_session_state DROP COLUMN IF EXISTS positions;
COMMIT;
