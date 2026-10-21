-- People, hierarchy and notification routing (Phase 1). Reversible: DROP TABLE the two tables.
-- Single-tenant today: tenant_id defaults to 'default' and is enforced on every query in
-- services/people so a second tenant works the day one exists.
CREATE TABLE IF NOT EXISTS proc.bp_person (
    subject            text        NOT NULL,            -- Cognito subject
    tenant_id          text        NOT NULL DEFAULT 'default',
    display_name       text,
    email              text,
    manager_subject    text,
    department_id      text,
    is_department_head boolean     NOT NULL DEFAULT false,
    roles              text[]      NOT NULL DEFAULT '{}',
    approval_limit     numeric,                         -- NULL = none, -1 = unlimited
    is_active          boolean     NOT NULL DEFAULT true,
    delegate_subject   text,
    delegate_until     timestamptz,
    last_modified_by   text,
    last_modified_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, subject)
);
CREATE INDEX IF NOT EXISTS ix_bp_person_manager ON proc.bp_person (tenant_id, manager_subject);
CREATE INDEX IF NOT EXISTS ix_bp_person_department ON proc.bp_person (tenant_id, department_id);

-- Which recipients each event type notifies; editable per tenant, 'default' tenant row is the fallback.
CREATE TABLE IF NOT EXISTS proc.bp_routing_rule (
    tenant_id        text  NOT NULL DEFAULT 'default',
    event_type       text  NOT NULL,
    targets          text[] NOT NULL,
    notify_actor     boolean NOT NULL DEFAULT false,
    last_modified_by text,
    last_modified_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, event_type)
);
