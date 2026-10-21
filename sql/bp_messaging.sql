-- User-to-user messaging (Phase 2). Reversible: DROP bp_message, bp_conversation_participant, bp_conversation.
CREATE TABLE IF NOT EXISTS proc.bp_conversation (
    conversation_id    bigserial PRIMARY KEY,
    tenant_id          text        NOT NULL DEFAULT 'default',
    type               text        NOT NULL CHECK (type IN ('direct','group','entity_thread')),
    title              text,
    direct_key         text,                       -- 'a|b' sorted, direct only: one conversation per pair
    linked_entity_type text,
    linked_entity_id   text,
    created_by         text        NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    last_message_at    timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_conversation_direct ON proc.bp_conversation (tenant_id, direct_key) WHERE type = 'direct';
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_conversation_entity ON proc.bp_conversation (tenant_id, linked_entity_type, linked_entity_id) WHERE type = 'entity_thread';

CREATE TABLE IF NOT EXISTS proc.bp_conversation_participant (
    conversation_id      bigint NOT NULL REFERENCES proc.bp_conversation ON DELETE CASCADE,
    user_subject         text   NOT NULL,
    role                 text   NOT NULL DEFAULT 'member' CHECK (role IN ('owner','member')),
    last_read_message_id bigint NOT NULL DEFAULT 0,
    muted                boolean NOT NULL DEFAULT false,
    joined_at            timestamptz NOT NULL DEFAULT now(),
    left_at              timestamptz,
    PRIMARY KEY (conversation_id, user_subject)
);
CREATE INDEX IF NOT EXISTS ix_bp_participant_user ON proc.bp_conversation_participant (user_subject) WHERE left_at IS NULL;

CREATE TABLE IF NOT EXISTS proc.bp_message (
    message_id      bigserial PRIMARY KEY,
    conversation_id bigint NOT NULL REFERENCES proc.bp_conversation ON DELETE CASCADE,
    sender_subject  text   NOT NULL,
    body            text   NOT NULL,
    mentions        text[] NOT NULL DEFAULT '{}',
    attachments     jsonb  NOT NULL DEFAULT '[]',
    reply_to_id     bigint,
    created_at      timestamptz NOT NULL DEFAULT now(),
    edited_at       timestamptz,
    deleted_at      timestamptz
);
CREATE INDEX IF NOT EXISTS ix_bp_message_conv ON proc.bp_message (conversation_id, message_id DESC);
