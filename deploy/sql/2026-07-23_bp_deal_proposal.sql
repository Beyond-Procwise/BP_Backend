BEGIN;
-- deploy/sql/2026-07-23_bp_deal_proposal.sql
-- 2026-07-23 Deal clustering proposals. A batch is clustered into PROPOSED sourcing
-- events; nothing existing is written. deal_id is minted only at confirm time, so it is
-- NULL on a proposal until a human accepts it. Idempotent / additive DDL only.

CREATE TABLE IF NOT EXISTS proc.bp_deal_proposal (
    proposal_id     BIGSERIAL PRIMARY KEY,
    batch_deal_id   VARCHAR NOT NULL,          -- the upload batch label it came from
    session_id      TEXT,
    proposed_name   VARCHAR,
    confidence      NUMERIC(5,2),              -- min pairwise F across members
    status          VARCHAR NOT NULL DEFAULT 'proposed',
                    -- proposed | confirmed | rejected | superseded
    deal_id         VARCHAR,                   -- NULL until confirm mints it
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    confirmed_at    TIMESTAMPTZ,
    confirmed_by    VARCHAR
);

CREATE TABLE IF NOT EXISTS proc.bp_deal_proposal_member (
    proposal_id     BIGINT NOT NULL REFERENCES proc.bp_deal_proposal(proposal_id) ON DELETE CASCADE,
    doc_type        VARCHAR NOT NULL,          -- quote | po | invoice
    doc_pk          VARCHAR NOT NULL,
    base_reference  VARCHAR,                   -- version-collapsed quote ref
    role            VARCHAR,                   -- anchor_quote | competing_quote | po | invoice
    match_score     NUMERIC(5,2),
    match_evidence  JSONB,                     -- score_link per-signal breakdown
    review_required BOOLEAN NOT NULL DEFAULT false,
    review_reasons  JSONB,
    PRIMARY KEY (proposal_id, doc_type, doc_pk)
);

CREATE INDEX IF NOT EXISTS ix_bp_deal_proposal_batch
    ON proc.bp_deal_proposal (batch_deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_deal_proposal_status
    ON proc.bp_deal_proposal (status);
CREATE INDEX IF NOT EXISTS ix_bp_deal_proposal_member_proposal
    ON proc.bp_deal_proposal_member (proposal_id);

COMMIT;
