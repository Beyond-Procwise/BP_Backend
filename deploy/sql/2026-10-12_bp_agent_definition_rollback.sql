-- Rollback for 2026-10-12_bp_agent_definition.sql.
-- DESTROYS every created agent and its history. Before running it, export
-- proc.bp_agent_definition (status = 1) back into agent_definitions.json, or
-- the agents people made are gone.
DROP TABLE IF EXISTS proc.bp_agent_definition_version;
DROP TABLE IF EXISTS proc.bp_agent_definition;
