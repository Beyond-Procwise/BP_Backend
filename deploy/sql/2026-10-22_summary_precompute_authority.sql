-- Who may warm the persona-summary cache on demand (POST /summary/precompute).
--
-- summary.precompute is a configure action, and the gate refuses those unless a policy states
-- effect=allow -- so until this row exists nobody can call it, an Admin included (and every Admin
-- attempt opens a "clarify policy" decision). Admin, because an explicit deal_ids list is honoured in
-- full and bypasses the scheduler's caps: one call can queue the shared model for hours. The nightly
-- scheduler runs the same job in-process with no principal and is not governed by this row.
-- FOR REVIEW. NOT APPLIED. Idempotent. Apply to BOTH bp_testdb and bp_sqldb.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'SummaryPrecomputeAuthorityPolicy', 'authority',
       'Who may warm the persona-summary cache on demand (a fleet-wide model job). No agent does this.',
       jsonb_build_object(
         'policy_identifier', 'summary_precompute_authority',
         'applies_to', jsonb_build_array('summary.precompute'),
         'required_role', 'Admin',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'migration:2026-10-22_summary_precompute_authority', now(), 'migration:2026-10-22_summary_precompute_authority'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'summary_precompute_authority'
       AND p.policy_status = 1);

COMMIT;
