-- NegotiationBoundsPolicy: the round limit, the round tactics and the final-offer phrases become governed.
--
-- These were constants in the negotiation agent (FINAL_OFFER_PATTERNS, a hard cap of 3 rounds, and the
-- figures compute_decision hardwired: gap report F22). The values added here are the ones the code ran
-- on, so behaviour is unchanged until someone edits the row.
--
-- Adds keys only: `new || existing` keeps every value the row already states, so re-running changes
-- nothing and a value someone has since edited is never overwritten. Reversible: _rollback.sql.
-- governed_limits RAISES on a missing value, so this is applied to a database BEFORE the code that reads it.

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}', $j${
     "max_rounds": 3,
     "round1_anchor_gap_pct": 0.10,
     "round2_gap_share": 0.6,
     "closing_risk_buffer_pct": 0.06,
     "closing_min_abs_buffer": 3.0,
     "closing_step_pct_of_gap": 0.12,
     "final_offer_phrases": [
       "best and final", "best final", "best offer we can", "best price we can", "cannot go lower",
       "final offer", "final price", "final quotation", "last price", "lowest we can do",
       "our best price", "rock bottom", "take it or leave it", "ultimatum"
     ]
   }$j$::jsonb || (policy_details->'rules'))
 WHERE policy_type = 'limit'
   AND policy_details->>'policy_identifier' = 'negotiation_bounds';
