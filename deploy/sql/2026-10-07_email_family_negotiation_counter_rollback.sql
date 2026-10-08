DELETE FROM proc.bp_policy
 WHERE policy_name = 'EmailFamily_negotiation_counter'
   AND created_by = 'email_assurance_migration';
