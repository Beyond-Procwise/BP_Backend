-- 2026-10-04  The builder snapshot a released report can be opened as.
--
-- proc.bp_report_job already stores two drawings of a report: the .pptx deck and the printable A4
-- page. This is the third — the JSON the report builder opens as editable pages, laid out on the
-- customer's own imported layouts.
--
-- jsonb, not bytea: it is read by an endpoint that serves JSON, and a status read must never haul
-- it (job_store's _COLUMNS already excludes the deck for that reason; `has_snapshot` is computed
-- the same way `has_page` is).
--
-- Idempotent. Safe to re-run.
BEGIN;

ALTER TABLE proc.bp_report_job
    ADD COLUMN IF NOT EXISTS snapshot jsonb;

-- WHICH PACK the pages were laid out on. A COLUMN, deliberately not part of the job's scope:
-- the scope is hashed into the Fact Pack's id (factpack.pack_id_for) and into the dedup key, so
-- carrying the style there would make one report with two packs two different Fact Packs — and
-- the style a report is drawn in does not change a single one of its facts.
ALTER TABLE proc.bp_report_job
    ADD COLUMN IF NOT EXISTS pack_key text;

COMMIT;
