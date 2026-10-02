-- ATB: a style pack and its page layouts, measured from an uploaded .pptx.
--
-- Design: specs/2026-10-02-pptx-style-import-design.md §4.
--
-- Candidates until a human approves them. `importing` is a status no read path serves:
-- get_conn() is AUTOCOMMIT, so an import cannot be one transaction, and a crash must leave
-- something invisible rather than something half-valid.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_style_pack (
    pack_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pack_key       text NOT NULL,
    version        integer NOT NULL,
    source_file    text NOT NULL,
    source_sha256  text NOT NULL,
    slide_count    integer NOT NULL,
    format         jsonb NOT NULL,
    tokens         jsonb NOT NULL,
    evidence       jsonb NOT NULL DEFAULT '{}'::jsonb,
    status         text NOT NULL DEFAULT 'importing'
                   CHECK (status IN ('importing', 'candidate', 'approved', 'rejected')),
    notes          text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text NOT NULL,
    approved_at    timestamptz,
    approved_by    text,
    CONSTRAINT uq_bp_style_pack_key_version UNIQUE (pack_key, version)
);

CREATE INDEX IF NOT EXISTS ix_bp_style_pack_status_created
    ON proc.bp_style_pack (status, created_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_page_layout (
    layout_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pack_id        uuid NOT NULL REFERENCES proc.bp_style_pack (pack_id) ON DELETE CASCADE,
    layout_key     text NOT NULL,
    proposed_name  text NOT NULL,
    name           text,
    slide_refs     integer[] NOT NULL DEFAULT '{}',
    regions        jsonb NOT NULL,
    slots          jsonb NOT NULL,
    example_fill   jsonb NOT NULL DEFAULT '{}'::jsonb,
    example_source jsonb NOT NULL DEFAULT '{}'::jsonb,
    problems       jsonb NOT NULL DEFAULT '[]'::jsonb,
    status         text NOT NULL DEFAULT 'candidate'
                   CHECK (status IN ('candidate', 'approved', 'rejected')),
    created_at     timestamptz NOT NULL DEFAULT now(),
    approved_at    timestamptz,
    approved_by    text,
    CONSTRAINT uq_bp_page_layout_pack_key UNIQUE (pack_id, layout_key)
);

CREATE INDEX IF NOT EXISTS ix_bp_page_layout_pack_status
    ON proc.bp_page_layout (pack_id, status);

COMMIT;
