BEGIN;
-- deploy/sql/2026-07-27_bp_style_feedback_rollback.sql
--
-- Reverses 2026-07-27_bp_style_feedback.sql.
--
-- Destroys the divergence history and any open recompile suggestions. The scores are
-- derived — they would rebuild from future drafts — but the record of which suggestions a
-- user already dismissed does not come back, so anyone who declined a recompile would be
-- asked again as soon as fresh evidence accumulated.
--
-- Suggestions are dropped before divergences: nothing links them, but the order keeps the
-- teardown readable as the mirror of the build.

DROP TABLE IF EXISTS proc.bp_style_recompile_suggestion;
DROP TABLE IF EXISTS proc.bp_style_divergence;

COMMIT;
