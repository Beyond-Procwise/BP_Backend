BEGIN;
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}', (policy_details->'rules') - 'rubric' - 'authority_agent')
 WHERE policy_name IN ('EmailFamily_negotiation_counter', 'EmailFamily_free_prompt') AND created_by = 'email_assurance_migration';
UPDATE proc.bp_policy
   SET policy_details = policy_details
       #- '{rules,fact_sources,supplier_current_offer,label}' #- '{rules,fact_sources,currency,label}'
       #- '{rules,fact_sources,supplier_lead_time,label}' #- '{rules,fact_sources,rfq_id,label}'
       #- '{rules,fact_sources,supplier_contact_name,label}'
 WHERE policy_name = 'EmailFamily_negotiation_counter' AND created_by = 'email_assurance_migration';
UPDATE proc.bp_policy
   SET policy_details = policy_details #- '{rules,fact_sources,supplier_contact_name,label}'
 WHERE policy_name = 'EmailFamily_free_prompt' AND created_by = 'email_assurance_migration';
COMMIT;
