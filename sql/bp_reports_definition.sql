-- SpendIQ Report builder (RB6) persistence: store the full builder STATE snapshot as a
-- JSONB "definition" on the existing proc.bp_reports table so saved report versions
-- survive across sessions. Legacy rows (report_url artifacts) keep definition NULL.
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS definition jsonb;
ALTER TABLE proc.bp_reports ADD COLUMN IF NOT EXISTS updated_ts timestamptz;
