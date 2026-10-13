-- Rollback of 2026-10-13_bp_prompt_render.sql.
-- Removes every stored render and the record of which run used which.
DROP TABLE IF EXISTS proc.bp_prompt_render_use;
DROP TABLE IF EXISTS proc.bp_prompt_render;
DROP FUNCTION IF EXISTS proc.bp_prompt_render_use_immutable();
