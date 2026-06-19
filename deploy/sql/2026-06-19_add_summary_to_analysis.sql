-- Store the deal's narrative summary text directly on its analysis-summary row,
-- so the metrics and the human-readable summary live together (authoritative
-- home for the deal narrative). The bp_summary row + narrative_summary_id FK are
-- still written for the existing summary ecosystem. Additive, idempotent.

ALTER TABLE proc.bp_analysis_summary
    ADD COLUMN IF NOT EXISTS summary TEXT;
