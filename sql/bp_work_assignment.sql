-- Who is working a piece of work (D-73 second half, 2026-10-22): a finding today, any item kind
-- later. One ACTIVE row per item; handing it on releases the old row and writes a new one, so the
-- history of hands stays. Reversible: DROP TABLE proc.bp_work_assignment.
CREATE TABLE IF NOT EXISTS proc.bp_work_assignment (
    assignment_id    bigserial PRIMARY KEY,
    tenant_id        text        NOT NULL DEFAULT 'default',
    entity_type      text        NOT NULL,          -- 'finding' (bp_extraction_discrepancy.discrepancy_id)
    entity_id        text        NOT NULL,
    assignee_subject text        NOT NULL,          -- the person it is with (Cognito sub)
    assigned_by      text        NOT NULL,
    assigned_at      timestamptz NOT NULL DEFAULT now(),
    note             text,
    released_at      timestamptz,                   -- set when handed on again or released
    released_by      text
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_bp_work_assignment_active
    ON proc.bp_work_assignment (tenant_id, entity_type, entity_id) WHERE released_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_bp_work_assignment_assignee
    ON proc.bp_work_assignment (tenant_id, assignee_subject) WHERE released_at IS NULL;
