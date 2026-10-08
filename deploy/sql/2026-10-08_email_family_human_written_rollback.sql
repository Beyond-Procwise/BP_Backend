DELETE FROM proc.bp_policy
 WHERE policy_name = 'EmailFamily_human_written'
   AND created_by = 'email_assurance_migration';
