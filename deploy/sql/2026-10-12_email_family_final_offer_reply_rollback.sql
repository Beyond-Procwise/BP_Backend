DELETE FROM proc.bp_policy
 WHERE policy_name = 'EmailFamily_final_offer_reply'
   AND created_by = 'email_assurance_migration';
