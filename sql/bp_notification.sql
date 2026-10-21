-- Domain-event outbox + per-user notifications (Phase 3). Reversible: DROP both tables.
CREATE TABLE IF NOT EXISTS proc.bp_domain_event (
    event_id     text PRIMARY KEY,                 -- caller-supplied or uuid; the idempotency key
    event_type   text        NOT NULL,
    tenant_id    text        NOT NULL DEFAULT 'default',
    actor_id     text,
    entity_type  text,
    entity_id    text,
    summary      text,
    data         jsonb       NOT NULL DEFAULT '{}',
    occurred_at  timestamptz NOT NULL DEFAULT now(),
    processed_at timestamptz,
    attempts     int         NOT NULL DEFAULT 0,
    last_error   text
);
CREATE INDEX IF NOT EXISTS ix_bp_domain_event_pending ON proc.bp_domain_event (occurred_at) WHERE processed_at IS NULL;

CREATE TABLE IF NOT EXISTS proc.bp_notification (
    notification_id bigserial PRIMARY KEY,
    tenant_id       text        NOT NULL DEFAULT 'default',
    event_id        text        NOT NULL,
    user_subject    text        NOT NULL,
    reason          text        NOT NULL,
    detail          text,                          -- "You're Jane's manager"
    title           text        NOT NULL,
    body            text,
    link            text,                          -- a UI route, never an API path
    priority        text        NOT NULL DEFAULT 'normal',
    read_at         timestamptz,
    channel_status  jsonb       NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (event_id, user_subject)                -- same event never notifies twice
);
CREATE INDEX IF NOT EXISTS ix_bp_notification_user ON proc.bp_notification (tenant_id, user_subject, read_at, created_at DESC);
