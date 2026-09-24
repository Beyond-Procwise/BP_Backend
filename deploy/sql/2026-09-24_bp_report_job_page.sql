-- 2026-09-24  Report jobs: the printable page beside the deck.
-- ---------------------------------------------------------------------------
-- Every released report is drawn twice from one composed AST: the .pptx deck and a
-- self-contained A4 HTML page (services/rga/render/html.py). The page is stored here,
-- held to the same rule as the deck -- only a released job carries one -- and a sign-off
-- binds both (grounding.deck_sha256 and grounding.page_sha256 in proc.bp_approval).
-- Jobs released before this carry no page. Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

ALTER TABLE proc.bp_report_job
    ADD COLUMN IF NOT EXISTS page            BYTEA,
    ADD COLUMN IF NOT EXISTS page_media_type TEXT;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_bp_report_job_page') THEN
    ALTER TABLE proc.bp_report_job
      ADD CONSTRAINT ck_bp_report_job_page CHECK (status = 'released' OR page IS NULL);
  END IF;
END $$;

COMMIT;
