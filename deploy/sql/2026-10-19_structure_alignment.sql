-- Brings bp_sqldb's structure in line with bp_testdb: eight tables the application creates on first use
-- (or that only scripts/tests create) and one column on each of two existing tables.
-- Definitions are generated from bp_testdb (columns, defaults, identities, constraints, indexes).
-- Idempotent; one transaction. bp_standin_actor_backfill and bp_supplier_id_crosswalk are included only to keep
-- the two databases identical during development.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_agent_group (
    group_id BIGSERIAL NOT NULL,
    name text NOT NULL,
    members jsonb NOT NULL,
    links jsonb NOT NULL DEFAULT '[]'::jsonb,
    is_active boolean NOT NULL DEFAULT true,
    created_by text,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    updated_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT bp_agent_group_pkey PRIMARY KEY (group_id)
);
CREATE INDEX IF NOT EXISTS ix_bp_agent_group_active ON proc.bp_agent_group USING btree (is_active, updated_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_egress_event (
    event_id BIGSERIAL NOT NULL,
    occurred_at timestamp with time zone NOT NULL,
    tenant_id text NOT NULL DEFAULT 'default'::text,
    purpose text NOT NULL,
    destination text NOT NULL,
    method text,
    outcome text NOT NULL,
    detail text,
    payload_sha256 text,
    payload_bytes integer,
    policy_version text,
    classification text[] NOT NULL DEFAULT '{}'::text[],
    CONSTRAINT bp_egress_event_pkey PRIMARY KEY (event_id)
);
CREATE INDEX IF NOT EXISTS ix_bp_egress_event_destination_occurred ON proc.bp_egress_event USING btree (destination, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_egress_event_purpose_occurred ON proc.bp_egress_event USING btree (purpose, occurred_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_egress_event_occurred_at ON proc.bp_egress_event USING btree (occurred_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_link_rejection (
    rejection_id BIGSERIAL NOT NULL,
    doc_type text NOT NULL,
    doc_pk text NOT NULL,
    po_id text NOT NULL,
    rejected_by text NOT NULL,
    note text,
    rejected_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT bp_link_rejection_doc_type_doc_pk_po_id_key UNIQUE (doc_type, doc_pk, po_id),
    CONSTRAINT bp_link_rejection_pkey PRIMARY KEY (rejection_id)
);
CREATE INDEX IF NOT EXISTS ix_bp_link_rejection_doc ON proc.bp_link_rejection USING btree (doc_type, doc_pk);

CREATE TABLE IF NOT EXISTS proc.email_dispatch (
    workflow_id text NOT NULL,
    supplier_id text,
    unique_id text NOT NULL,
    round integer NOT NULL,
    message_id text NOT NULL,
    dispatched_at timestamp with time zone NOT NULL,
    subject text,
    status text NOT NULL DEFAULT 'sent'::text,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    updated_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT email_dispatch_pkey PRIMARY KEY (workflow_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_email_dispatch_workflow ON proc.email_dispatch USING btree (workflow_id);
CREATE INDEX IF NOT EXISTS idx_email_dispatch_unique ON proc.email_dispatch USING btree (unique_id);

CREATE TABLE IF NOT EXISTS proc.supplier_risk_scores (
    supplier_id text NOT NULL,
    score numeric(6,3) NOT NULL,
    model_version text NOT NULL,
    feature_summary jsonb NOT NULL,
    computed_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT supplier_risk_scores_pkey PRIMARY KEY (supplier_id)
);

CREATE TABLE IF NOT EXISTS proc.supplier_risk_signals (
    id SERIAL NOT NULL,
    supplier_id text NOT NULL,
    signal_type text NOT NULL,
    severity numeric(5,2) NOT NULL,
    source text NOT NULL,
    payload jsonb,
    occurred_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT supplier_risk_signals_pkey PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS idx_supplier_risk_signals_supplier ON proc.supplier_risk_signals USING btree (supplier_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_standin_actor_backfill (
    backfill_id BIGSERIAL NOT NULL,
    table_name text NOT NULL,
    row_key text NOT NULL,
    column_name text NOT NULL,
    old_value text NOT NULL,
    cleared_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT bp_standin_actor_backfill_pkey PRIMARY KEY (backfill_id)
);
CREATE INDEX IF NOT EXISTS ix_bp_standin_actor_backfill_table ON proc.bp_standin_actor_backfill USING btree (table_name, column_name);

CREATE TABLE IF NOT EXISTS proc.bp_supplier_id_crosswalk (
    bp_supplier_id text NOT NULL,
    uicanvas_supplier_id text NOT NULL,
    legal_entity_key text NOT NULL,
    created_date timestamp without time zone NOT NULL DEFAULT now(),
    CONSTRAINT bp_supplier_id_crosswalk_uicanvas_supplier_id_key UNIQUE (uicanvas_supplier_id),
    CONSTRAINT bp_supplier_id_crosswalk_pkey PRIMARY KEY (bp_supplier_id)
);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_id_crosswalk_legal_entity_key ON proc.bp_supplier_id_crosswalk USING btree (legal_entity_key);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_id_crosswalk_uicanvas_supplier_id ON proc.bp_supplier_id_crosswalk USING btree (uicanvas_supplier_id);

ALTER TABLE proc.bp_workflow_run ADD COLUMN IF NOT EXISTS initiated_by TEXT;
ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS positions JSONB;

COMMIT;
