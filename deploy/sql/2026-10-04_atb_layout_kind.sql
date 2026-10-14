-- 2026-10-04  What an imported layout IS: a reusable template, or one page someone arranged.
--
-- The step 1 design ruled (§6a) that a deck's single-use slides "wait for step 2 and are then
-- imported as composed pages": a quadrant is not a template, and a Layout picker holding 26
-- near-identical skeletons is the original complaint in a new form.
--
-- Stated in a column rather than inferred from slide_refs having one element. That inference is
-- true today only because cluster.reused is defined as len(slides) > 1, and coupling a product
-- distinction to that definition means a change to the clustering silently reclassifies stored
-- rows.
--
-- Defaults to 'template', so every row written before today keeps the meaning it had.
--
-- Idempotent. Safe to re-run.

BEGIN;

ALTER TABLE proc.bp_page_layout
    ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'template';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_bp_page_layout_kind') THEN
        ALTER TABLE proc.bp_page_layout
            ADD CONSTRAINT ck_bp_page_layout_kind CHECK (kind IN ('template', 'page'));
    END IF;
END $$;

-- The Papers screen and the pickers both ask for one kind at a time.
CREATE INDEX IF NOT EXISTS ix_bp_page_layout_pack_kind
    ON proc.bp_page_layout (pack_id, kind, status);

COMMIT;
