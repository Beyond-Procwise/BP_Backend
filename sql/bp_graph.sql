-- People graph support (Phase 4). Reversible: DROP bp_activity, bp_proposed_edge; ALTER TABLE ... DROP COLUMN title.
ALTER TABLE proc.bp_person ADD COLUMN IF NOT EXISTS title text;

-- Observed behaviour. Metadata only, never message content. Not the graph.
CREATE TABLE IF NOT EXISTS proc.bp_activity (
    id              bigserial PRIMARY KEY,
    tenant_id       text        NOT NULL DEFAULT 'default',
    actor_id        text        NOT NULL,
    verb            text        NOT NULL CHECK (verb IN ('message.sent','mention','comment','view','edit',
                        'suggestion.shown','suggestion.accepted','suggestion.dismissed',
                        'notification.opened','notification.ignored')),
    target_user_id  text,
    entity_type     text,
    entity_id       text,
    conversation_id bigint,
    occurred_at     timestamptz NOT NULL DEFAULT now(),
    duration_ms     bigint,
    metadata        jsonb       NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_bp_activity_actor  ON proc.bp_activity (tenant_id, actor_id, occurred_at);
CREATE INDEX IF NOT EXISTS ix_bp_activity_entity ON proc.bp_activity (tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_bp_activity_target ON proc.bp_activity (tenant_id, target_user_id, occurred_at);

-- Inferences (e.g. from embeddings) wait here until an admin or the entity owner confirms them.
CREATE TABLE IF NOT EXISTS proc.bp_proposed_edge (
    id            bigserial PRIMARY KEY,
    tenant_id     text        NOT NULL DEFAULT 'default',
    src_label     text        NOT NULL,
    src_key       text        NOT NULL,
    edge_type     text        NOT NULL,
    dst_label     text        NOT NULL,
    dst_key       text        NOT NULL,
    props         jsonb       NOT NULL DEFAULT '{}',
    confidence    numeric     NOT NULL,
    model_version text        NOT NULL,
    evidence      jsonb       NOT NULL DEFAULT '{}',
    status        text        NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed','confirmed','rejected')),
    proposed_at   timestamptz NOT NULL DEFAULT now(),
    decided_by    text,
    decided_at    timestamptz,
    UNIQUE (tenant_id, src_label, src_key, edge_type, dst_label, dst_key, model_version)
);
CREATE INDEX IF NOT EXISTS ix_bp_proposed_edge_status ON proc.bp_proposed_edge (tenant_id, status, proposed_at);
