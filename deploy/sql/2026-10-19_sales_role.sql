-- The Sales role, approved 2026-10-19: a role definition plus a group mapping. Applied to bp_testdb first; promote to bp_sqldb
-- after review (both databases hold the policy rows).
--
-- LEAST PRIVILEGE, DELIBERATELY. Sales is rank 1 and may only read and compute.
--   * Rank, not name, is what policies compare: a policy that names a required role denies anyone BELOW that rank. At rank 2 (Buyer's)
--     a sales user would pass the Buyer-level permits for supplier master writes, document upload and promotion, email send and
--     approval, web research and prompt authoring.
--   * No `write` class: it is the broad class (contracts, deals, findings, catalog, accounts) and no per-action policy blocks most
--     of it. Drafting quotes through the API (sales_quote.create is class write) therefore stays with Buyer and above until a
--     deliberate decision gives Sales a narrower write path.
--   * Margin: margin.read requires Buyer rank (SalesMarginReadAuthorityPolicy), so Sales cannot read cost or margin. Widening that
--     is a separate decision about that one policy, not a side effect of this role.
-- What Sales CAN do: pass the gateway's role rule for sales quotes (get_quote, list_quotes), because quote.read is a read.
--
-- Group: bp-sales (the existing bp-* naming). A user in several groups still resolves by multiple_groups: highest_rank.
-- Idempotent: each statement is a no-op once the entry exists. Reversible: 2026-10-19_sales_role_rollback.sql.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,roles}',
           COALESCE(policy_details #> '{rules,roles}', '{}'::jsonb)
           || '{"Sales": {"rank": 1, "allow": ["read", "compute"]}}'::jsonb, true),
       version = version + 1,
       last_modified_by = 'sales_role',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_definition'
   AND NOT ((policy_details #> '{rules,roles}') ? 'Sales');

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules,group_to_role}',
           COALESCE(policy_details #> '{rules,group_to_role}', '{}'::jsonb) || '{"bp-sales": "Sales"}'::jsonb, true),
       version = version + 1,
       last_modified_by = 'sales_role',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'role_assignment'
   AND NOT ((policy_details #> '{rules,group_to_role}') ? 'bp-sales');

COMMIT;
