-- Extraction feedback loop v1: per-vendor hint proposal queue.
-- The queue of PROPOSED extraction hints awaiting human approval. On approval,
-- apply.py writes a versioned proc.bp_prompt row (prompt_type='extraction_vendor_hint');
-- this table only holds the pending/reviewed proposals + their evidence.
-- Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS proc.bp_extraction_hint_proposal (
    proposal_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_date        TIMESTAMPTZ NOT NULL DEFAULT now(),
    doc_type            TEXT        NOT NULL,          -- invoice | quote | purchase_order
    vendor_key          TEXT        NOT NULL,          -- normalized vendor hint (vendor_key.py)
    field_name          TEXT,                          -- specific field, or NULL for a general vendor hint
    dedup_key           TEXT        NOT NULL,          -- hash(doc_type,vendor_key,field,failure_signature)
    evidence            JSONB       NOT NULL,          -- {doc_ids[], sample_count, missing_count, failure_rate, snippets[]}
    proposed_hint       TEXT        NOT NULL,          -- human-readable hint text
    rationale           TEXT,
    status              TEXT        NOT NULL DEFAULT 'pending',  -- pending|approved|rejected|superseded
    reviewed_by         TEXT,
    reviewed_date       TIMESTAMPTZ,
    review_reason       TEXT,
    resulting_prompt_id BIGINT                          -- proc.bp_prompt.prompt_id created on approve
);

-- Only one live proposal per scope+signature (pending or already approved).
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_ext_hint_proposal_dedup
    ON proc.bp_extraction_hint_proposal (dedup_key)
    WHERE status IN ('pending', 'approved');

CREATE INDEX IF NOT EXISTS ix_bp_ext_hint_proposal_pending
    ON proc.bp_extraction_hint_proposal (status) WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS ix_bp_ext_hint_proposal_scope
    ON proc.bp_extraction_hint_proposal (doc_type, vendor_key);
