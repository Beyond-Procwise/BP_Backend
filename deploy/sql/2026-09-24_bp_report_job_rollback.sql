-- Drops the report job table, and with it every stored deck. The audit events
-- in bp_agent_actions are untouched: a report's run stays traceable, only the
-- file is gone.
BEGIN;

DROP TABLE IF EXISTS proc.bp_report_job;

COMMIT;
