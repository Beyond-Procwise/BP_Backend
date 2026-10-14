-- Rollback of 2026-10-13_bp_prompt_registry.sql.
-- Removes the registry, every saved version and the whole promotion history.
-- proc.bp_prompt was never altered, so the older readers are unaffected.
DROP TABLE IF EXISTS proc.bp_prompt_pointer_move;
DROP TABLE IF EXISTS proc.bp_prompt_version;
DROP TABLE IF EXISTS proc.bp_prompt_registry;
DROP FUNCTION IF EXISTS proc.bp_prompt_pointer_move_immutable();
DROP FUNCTION IF EXISTS proc.bp_prompt_version_immutable();
DROP FUNCTION IF EXISTS proc.bp_prompt_registry_guard();
