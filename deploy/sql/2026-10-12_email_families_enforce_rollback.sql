UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,mode}', '"shadow"')
 WHERE policy_type = 'email_family'
   AND created_by = 'email_assurance_migration'
   AND policy_details->'rules'->>'mode' = 'enforce';
