-- 2026-06-15 Quote-anchored: trace each opportunity to its anchoring quote.
ALTER TABLE proc.bp_opportunity ADD COLUMN IF NOT EXISTS quote_id VARCHAR;
ALTER TABLE proc.bp_opportunity ADD COLUMN IF NOT EXISTS po_id    VARCHAR;
CREATE INDEX IF NOT EXISTS ix_bp_opportunity_quote ON proc.bp_opportunity (quote_id);
