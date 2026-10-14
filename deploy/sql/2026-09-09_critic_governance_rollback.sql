BEGIN;
DELETE FROM proc.bp_policy WHERE policy_name = 'opportunity_critic_thresholds';
DELETE FROM proc.bp_prompt WHERE prompt_name = 'opportunity_critic_system';
COMMIT;
