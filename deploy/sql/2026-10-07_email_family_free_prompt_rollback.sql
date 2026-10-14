DELETE FROM proc.bp_policy
 WHERE policy_name = 'EmailFamily_free_prompt'
   AND created_by = 'email_assurance_migration';
