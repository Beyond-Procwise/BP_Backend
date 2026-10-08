-- deploy/sql/2026-10-08_contract_parent_thresholds_rollback.sql
-- Only safe together with reverting the code: the code RAISES without these keys.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}',
         (policy_details -> 'rules') - 'contract_parent_min_score' - 'contract_parent_separation'),
       last_modified_date = NOW(),
       last_modified_by   = 'deploy/sql/2026-10-08_contract_parent_thresholds_rollback.sql',
       version            = COALESCE(version, 1) + 1
 WHERE policy_details ->> 'policy_identifier' = 'promotion_thresholds';
