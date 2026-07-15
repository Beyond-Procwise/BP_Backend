-- deploy/sql/2026-07-15_bp_deal.sql
-- 2026-07-15 Deal header: draft vs tracked lifecycle for uploaded analyses.
-- A new upload starts as a DRAFT (is_tracked=false); Promote flips it tracked.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_deal (
    deal_id             text PRIMARY KEY,
    is_tracked          boolean NOT NULL DEFAULT false,
    is_saved_reference  boolean NOT NULL DEFAULT false,
    tracked_at          timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_deal_is_tracked ON proc.bp_deal (is_tracked);

-- One-time backfill: every deal that already exists stays visible in Pipeline.
INSERT INTO proc.bp_deal (deal_id, is_tracked, tracked_at)
SELECT deal_id, true, now()
  FROM proc.bp_deal_overview
 WHERE deal_id IS NOT NULL
ON CONFLICT (deal_id) DO NOTHING;

COMMIT;
