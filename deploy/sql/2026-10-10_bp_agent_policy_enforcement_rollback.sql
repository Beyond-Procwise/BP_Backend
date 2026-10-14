BEGIN;
DROP INDEX IF EXISTS proc.ix_bp_decision_open_respond_by;
ALTER TABLE proc.bp_decision
    DROP COLUMN IF EXISTS options, DROP COLUMN IF EXISTS respond_by, DROP COLUMN IF EXISTS on_timeout,
    DROP COLUMN IF EXISTS decision_scope, DROP COLUMN IF EXISTS levels, DROP COLUMN IF EXISTS current_level;
DROP TABLE IF EXISTS proc.bp_policy_notification;
DROP TRIGGER IF EXISTS trg_bp_policy_firing_guard ON proc.bp_policy_firing;
DROP TABLE IF EXISTS proc.bp_policy_firing;
DROP FUNCTION IF EXISTS proc.bp_policy_firing_guard();
DROP TABLE IF EXISTS proc.bp_policy_decider_map;
COMMIT;
