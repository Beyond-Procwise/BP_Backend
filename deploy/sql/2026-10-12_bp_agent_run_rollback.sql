-- Rollback for 2026-10-12_bp_agent_run.sql. DESTROYS the record of every
-- reasoning run; nothing else reads the table.
DROP TABLE IF EXISTS proc.bp_agent_run;
