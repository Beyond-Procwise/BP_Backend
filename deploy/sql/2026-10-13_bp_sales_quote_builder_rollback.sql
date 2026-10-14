-- Rollback for 2026-10-13_bp_sales_quote_builder.sql.
--
-- Safe only while no builder quote exists. A quote with no account cannot go back
-- under NOT NULL, so this stops rather than deleting one: remove or re-home those
-- quotes first, by hand, knowing what they are.

BEGIN;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM proc.bp_sales_quote WHERE account_id IS NULL) THEN
        RAISE EXCEPTION 'quotes with no account exist; this rollback would lose them';
    END IF;
END $$;

DROP INDEX IF EXISTS proc.ix_bp_sales_quote_deal_live;

ALTER TABLE proc.bp_sales_quote_internal
    DROP COLUMN IF EXISTS costed_lines,
    DROP COLUMN IF EXISTS uncosted_lines;

ALTER TABLE proc.bp_sales_quote_line_internal
    DROP COLUMN IF EXISTS cost_source,
    DROP COLUMN IF EXISTS markup_pct;

ALTER TABLE proc.bp_sales_quote_line
    DROP COLUMN IF EXISTS line_tag,
    DROP COLUMN IF EXISTS entered_discount_pct;

ALTER TABLE proc.bp_sales_quote
    DROP COLUMN IF EXISTS deal_id,
    DROP COLUMN IF EXISTS version,
    DROP COLUMN IF EXISTS customer_name,
    DROP COLUMN IF EXISTS customer_contact,
    DROP COLUMN IF EXISTS category,
    DROP COLUMN IF EXISTS prepared_by,
    DROP COLUMN IF EXISTS scope_html,
    DROP COLUMN IF EXISTS assumptions_html,
    DROP COLUMN IF EXISTS terms,
    DROP COLUMN IF EXISTS conditions,
    ALTER COLUMN account_id SET NOT NULL;

COMMIT;
