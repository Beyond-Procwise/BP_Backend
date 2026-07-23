BEGIN;
-- deploy/sql/2026-07-23_bp_deal_proposal_flags.sql
-- Proposal-level HITL flags (e.g. "unresolved supplier on SDP-Q-44120") were computed
-- by cluster_batch but dropped by store_proposals -- the reviewer never saw them
-- (spec Error handling). Idempotent / additive DDL only.

ALTER TABLE proc.bp_deal_proposal ADD COLUMN IF NOT EXISTS flags JSONB;

COMMIT;
