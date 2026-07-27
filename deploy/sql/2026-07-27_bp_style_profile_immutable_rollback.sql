BEGIN;
-- deploy/sql/2026-07-27_bp_style_profile_immutable_rollback.sql
--
-- Reverses 2026-07-27_bp_style_profile_immutable.sql.
--
-- Running this makes approved profiles editable again. The rows survive, but invariant 5
-- stops being enforced anywhere except in application code, so a direct UPDATE could
-- silently change the rules a past draft was generated under while the draft continues to
-- cite that version as its provenance.

DROP TRIGGER IF EXISTS tr_bp_style_profile_freeze_approved ON proc.bp_style_profile;
DROP FUNCTION IF EXISTS proc.bp_style_profile_freeze_approved();

COMMIT;
