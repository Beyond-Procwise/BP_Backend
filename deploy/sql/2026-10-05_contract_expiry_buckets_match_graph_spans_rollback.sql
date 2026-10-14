UPDATE proc.bp_rule
   SET conditions = jsonb_set(conditions, '{bucket_months}', '[3,6,9,12,18]'::jsonb),
       version = version + 1, last_modified_date = now()
 WHERE detector_slug = 'contract_expiry_bucket_check';
