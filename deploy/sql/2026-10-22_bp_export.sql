-- Exports: the files an agent builds for a person on request (a Word note, a spreadsheet, a deck, a chart).
-- FOR REVIEW. NOT APPLIED to any database. Idempotent; one transaction. Rollback: the _rollback.sql beside it.
--
-- Why a table: the agent tools build_docx / build_xlsx / build_pptx / render_chart produce bytes inside the
-- agent's process, and the person collects the file later from the Exports list (GET /exports/{export_id}).
-- A row holds the file, who it was built for, and what it was built from; it is never a business record
-- (nothing in it is a contract, an order or an approval), so there is no append-only trigger: a person may
-- delete their own export. Content is capped by the builders (EXPORT_MAX_BYTES), not here.
-- tenant_id follows the tenancy convention (deploy/sql/tenancy): 'default' until the stamping trigger replaces it.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_export (
    export_id     TEXT PRIMARY KEY,
    kind          TEXT NOT NULL
                  CONSTRAINT bp_export_kind_check CHECK (kind IN ('docx', 'xlsx', 'pptx', 'svg')),
    filename      TEXT NOT NULL,
    media_type    TEXT NOT NULL,
    title         TEXT NOT NULL,
    content       BYTEA NOT NULL,
    size_bytes    INTEGER NOT NULL,
    source        TEXT NOT NULL DEFAULT 'agent',      -- what built it: an agent tool, a skill name, a report
    created_by    TEXT NOT NULL,                      -- the person the file was built for
    created_date  TIMESTAMPTZ NOT NULL DEFAULT now(),
    tenant_id     TEXT NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS ix_bp_export_owner_recent
    ON proc.bp_export (created_by, created_date DESC);

COMMIT;
