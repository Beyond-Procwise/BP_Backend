-- A negotiation remembers the deal it is about.
--
-- A supplier's emailed reply reaches the NegotiationAgent with no deal or quote
-- on it, so a reply round had no Advisor posture and no plays. The agent now
-- saves the deal with the session on the round that names it, and reads it back
-- on rounds that do not. Existing sessions stay NULL: none of them can be
-- traced to a deal through any other table (checked 2026-10-15), and a deal is
-- never guessed. Re-runnable.

ALTER TABLE proc.negotiation_session_state ADD COLUMN IF NOT EXISTS deal_id TEXT;

COMMENT ON COLUMN proc.negotiation_session_state.deal_id IS
  'The deal this negotiation is about, saved by the NegotiationAgent from the round that named it (or its quote). NULL when no round has named one.';
