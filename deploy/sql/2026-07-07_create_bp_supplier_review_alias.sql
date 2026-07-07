-- Supplier entity-resolution: review queue + alias map. Idempotent.
-- The document's literal supplier value is NEVER rewritten; these tables govern
-- only the canonical supplier_id link and surface ambiguous identity decisions.

-- Alias map: a variant name -> canonical supplier_id. Checked FIRST by the
-- resolver, so confirmed/rejected variants resolve deterministically and are
-- never re-flagged.
CREATE TABLE IF NOT EXISTS proc.bp_supplier_alias (
    alias_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    alias_name    TEXT        NOT NULL,
    supplier_id   TEXT        NOT NULL,
    created_by    TEXT        NOT NULL DEFAULT 'system',
    created_date  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_supplier_alias_name
    ON proc.bp_supplier_alias (LOWER(alias_name));

-- Review queue: close-call supplier-identity decisions awaiting human confirm.
CREATE TABLE IF NOT EXISTS proc.bp_supplier_review (
    review_id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_date            TIMESTAMPTZ NOT NULL DEFAULT now(),
    extracted_name          TEXT        NOT NULL,   -- literal value from the document
    decision                TEXT        NOT NULL,   -- linked | created (the auto-decision)
    chosen_supplier_id      TEXT,                   -- supplier_id the auto-decision picked
    candidate_supplier_id   TEXT,                   -- the near-match it was close to
    candidate_supplier_name TEXT,
    score                   NUMERIC,                -- fuzzy WRatio (0-100)
    doc_pk                  TEXT,
    doc_type                TEXT,
    trace_id                TEXT,
    status                  TEXT        NOT NULL DEFAULT 'pending',  -- pending|confirmed|rejected
    reviewed_by             TEXT,
    reviewed_date           TIMESTAMPTZ
);
-- One live flag per (name, candidate) pair.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_supplier_review_pending
    ON proc.bp_supplier_review (LOWER(extracted_name), candidate_supplier_id)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS ix_bp_supplier_review_status
    ON proc.bp_supplier_review (status) WHERE status = 'pending';
