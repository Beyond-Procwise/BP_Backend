-- 2026-10-04  The index the composer actually uses.
--
-- ix_bp_page_layout_pack_kind is (pack_id, kind, status), which serves the Papers screen — it
-- opens ONE paper and asks for its layouts. But the call every composer load makes is
-- GET /atb/layouts?status=approved, which passes NO pack_id, so a leading pack_id column cannot
-- be used for it at all: that query is the one atbLoadApproved makes to fill both registries.
--
-- Added rather than replacing the other: the per-pack lookup is still a real query, and both
-- indexes are small.
--
-- Idempotent. Safe to re-run.

BEGIN;

CREATE INDEX IF NOT EXISTS ix_bp_page_layout_status_kind
    ON proc.bp_page_layout (status, kind);

COMMIT;
