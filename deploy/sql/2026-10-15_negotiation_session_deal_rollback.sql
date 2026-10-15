-- Rollback for 2026-10-15_negotiation_session_deal.sql. The agent reads and
-- writes the column only when it exists, so it keeps working without it.

ALTER TABLE proc.negotiation_session_state DROP COLUMN IF EXISTS deal_id;
