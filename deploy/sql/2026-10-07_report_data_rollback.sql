DROP INDEX IF EXISTS proc.ix_bp_reports_data_mode;
ALTER TABLE proc.bp_reports DROP CONSTRAINT IF EXISTS bp_reports_data_mode_chk;
ALTER TABLE proc.bp_reports DROP COLUMN IF EXISTS data_mode;
DROP TABLE IF EXISTS proc.bp_presentation_log;
DROP TABLE IF EXISTS proc.bp_user_buyer_scope;
