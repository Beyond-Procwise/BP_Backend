-- Rollback for 2026-10-15_ask_screen_agents.sql. Safe at any time: with the row
-- gone every screen answers as General, which is the Ask box before this change.

BEGIN;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'ask_screen_agents'
   AND created_by = 'ask_routing';

COMMIT;
