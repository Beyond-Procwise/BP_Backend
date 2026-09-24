-- The code that reads these tables must be rolled back first.
BEGIN;
DROP TABLE IF EXISTS proc.bp_user_preference;
DROP TABLE IF EXISTS proc.bp_translation;
COMMIT;
