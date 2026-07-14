-- Contract obligation intelligence: grounded n-ary obligations read from contract prose.
-- Spec: docs/superpowers/specs/2026-07-14-contract-obligation-intelligence-design.md

CREATE TABLE IF NOT EXISTS proc.bp_contract_obligation (
    obligation_id   bigserial PRIMARY KEY,
    document_id     text NOT NULL,
    contract_id     text,
    name            text NOT NULL,
    obligation_type text NOT NULL,           -- schema.RelType
    clause_ref      text,
    -- Verbatim span from the source document. The grounding guard has already proven this
    -- string occurs in the contract; an obligation without it is not a fact.
    source_quote    text NOT NULL,
    grounded        boolean NOT NULL DEFAULT TRUE,
    created_date    timestamptz NOT NULL DEFAULT now()
);

-- The party table is what makes a hyperedge queryable: one obligation binds MANY entities,
-- so "which obligations involve supplier X and a penalty?" is a join, not a graph traversal.
CREATE TABLE IF NOT EXISTS proc.bp_contract_obligation_party (
    obligation_id bigint NOT NULL
        REFERENCES proc.bp_contract_obligation (obligation_id) ON DELETE CASCADE,
    entity_name   text NOT NULL,
    entity_type   text NOT NULL,             -- schema.EntityType
    PRIMARY KEY (obligation_id, entity_name)
);

-- Without this, "we read the contract and it has no obligations" and "we never managed to
-- read the contract" both surface as an empty list — a green zero the system has not earned.
CREATE TABLE IF NOT EXISTS proc.bp_contract_obligation_run (
    document_id  text PRIMARY KEY,
    status       text NOT NULL,             -- extracted | no_grounded_obligations | failed
    n_grounded   integer NOT NULL DEFAULT 0,
    n_dropped    integer NOT NULL DEFAULT 0,
    error        text,
    created_date timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_contract_obligation_document
    ON proc.bp_contract_obligation (document_id);
CREATE INDEX IF NOT EXISTS ix_bp_contract_obligation_party_entity
    ON proc.bp_contract_obligation_party (entity_name);
