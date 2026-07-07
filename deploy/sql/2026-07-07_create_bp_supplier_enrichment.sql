-- Web-enabled AgentNick supplier research: provenance-stamped enrichment sidecar.
-- Never overwrites bp_supplier; only records researched facts + citations, and
-- which empty fields were auto-applied. Idempotent.
CREATE TABLE IF NOT EXISTS proc.bp_supplier_enrichment (
    enrichment_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_date   TIMESTAMPTZ NOT NULL DEFAULT now(),
    supplier_id    TEXT        NOT NULL,
    source         TEXT        NOT NULL DEFAULT 'agentnick_web',
    model          TEXT,
    fields         JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- {field:{value,source_url,confidence}}
    citations      JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- all URLs the tools returned/used
    confidence     NUMERIC,
    raw            JSONB,
    apply_status   TEXT        NOT NULL DEFAULT 'pending',      -- pending|applied|rejected
    applied_fields JSONB       NOT NULL DEFAULT '{}'::jsonb,
    reviewed_by    TEXT,
    reviewed_date  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_enrichment_sup
    ON proc.bp_supplier_enrichment (supplier_id, created_date DESC);
