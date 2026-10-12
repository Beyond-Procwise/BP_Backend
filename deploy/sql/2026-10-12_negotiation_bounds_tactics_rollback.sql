UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}',
         (policy_details->'rules') - 'max_rounds' - 'round1_anchor_gap_pct' - 'round2_gap_share'
           - 'closing_risk_buffer_pct' - 'closing_min_abs_buffer' - 'closing_step_pct_of_gap' - 'final_offer_phrases')
 WHERE policy_type = 'limit'
   AND policy_details->>'policy_identifier' = 'negotiation_bounds';
