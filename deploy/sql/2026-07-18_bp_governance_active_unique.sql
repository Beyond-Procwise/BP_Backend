-- One ACTIVE governance row per (type, name).
--
-- Why this is needed now: the Prompts and Policies admin screens have been repointed off
-- uicanvas.proc.prompt|policy (which no agent reads) onto proc.bp_prompt|bp_policy (which
-- every agent reads). That makes those screens a real control surface, and it also makes
-- them able to create the one thing the resolvers cannot cope with: two active rows for the
-- same (type, name).
--
-- Every governance resolver looks up a single row and takes the first match, e.g.
--   SELECT prompts_desc FROM proc.bp_prompt
--    WHERE prompt_type = 'ask_persona' AND prompt_name = 'joshi'
--      AND COALESCE(prompts_status, 1) = 1 LIMIT 1
-- With two active rows, WHICH prompt governs the assistant is decided by physical row order.
-- ORDER BY clauses were added to the engines as a stopgap; this makes the ambiguity
-- impossible to create in the first place, which is the only real fix.
--
-- Partial (WHERE status = 1) on purpose: superseded rows are the version history. The update
-- path writes a new active row and flips the previous one to 0, so history accumulates
-- freely while exactly one row per (type, name) is ever live.
--
-- Verified before applying: zero active duplicates and zero NULL types in either table.

CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_prompt_active_type_name
    ON proc.bp_prompt (prompt_type, prompt_name)
 WHERE prompts_status = 1;

CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_policy_active_type_name
    ON proc.bp_policy (policy_type, policy_name)
 WHERE policy_status = 1;
