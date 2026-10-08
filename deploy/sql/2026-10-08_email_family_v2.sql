-- Family rows v2: a human label per fact, a rubric for the judge, and the agent whose
-- authority a counter is checked against.
--
-- Updates ONLY the two rows created by email_assurance_migration; touches no other policy.
-- Idempotent (each UPDATE is guarded by the key it adds). Reversible: _rollback.sql.
-- NOT APPLIED to bp_sqldb.

BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}', (policy_details->'rules') || $j${
     "rubric": ["ask_is_specific", "position_follows_from_offer", "deadline_stated", "tone_matches_escalation_level", "concise"],
     "authority_agent": "email_drafting_agent"
   }$j$::jsonb)
 WHERE policy_name = 'EmailFamily_negotiation_counter' AND created_by = 'email_assurance_migration'
   AND NOT (policy_details->'rules' ? 'rubric');

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(jsonb_set(jsonb_set(jsonb_set(jsonb_set(policy_details,
         '{rules,fact_sources,supplier_current_offer,label}', '"Supplier''s latest offer"'),
         '{rules,fact_sources,currency,label}', '"Currency of the offer"'),
         '{rules,fact_sources,supplier_lead_time,label}', '"Supplier''s stated lead time"'),
         '{rules,fact_sources,rfq_id,label}', '"RFQ reference"'),
         '{rules,fact_sources,supplier_contact_name,label}', '"Supplier contact"')
 WHERE policy_name = 'EmailFamily_negotiation_counter' AND created_by = 'email_assurance_migration'
   AND NOT (policy_details->'rules'->'fact_sources'->'currency' ? 'label');

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(jsonb_set(policy_details, '{rules}', (policy_details->'rules') || $j${
     "rubric": ["completeness", "clarity_of_ask", "tone_fit", "concision"]
   }$j$::jsonb), '{rules,fact_sources,supplier_contact_name,label}', '"Supplier contact"')
 WHERE policy_name = 'EmailFamily_free_prompt' AND created_by = 'email_assurance_migration'
   AND NOT (policy_details->'rules' ? 'rubric');

COMMIT;
