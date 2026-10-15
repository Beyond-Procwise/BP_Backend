-- Rollback for 2026-10-15_margin_floor_enforcement.sql. With the rules gone the code reads
-- the strict defaults (refuse / refuse / false); with the override row gone nobody may override.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}',
         (policy_details->'rules') - 'margin_floor_action' - 'margin_floor_untested'
                                   - 'margin_floor_applies_to_catalog'),
       last_modified_date = now(), last_modified_by = 'sales_quote_margin_floor'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing';

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'sales_quote_margin_floor_override';

COMMIT;
