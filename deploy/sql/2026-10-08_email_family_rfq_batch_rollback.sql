DELETE FROM proc.bp_policy
 WHERE policy_name = 'EmailFamily_rfq_batch'
   AND created_by = 'email_assurance_migration';
