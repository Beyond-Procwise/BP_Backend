-- P6: the workflows that may run without a governance envelope, stated in policy.
--
-- Until now this was one line in orchestrator.py:
--
--     if workflow_name == "document_extraction":
--         return None
--
-- The exemption is correct -- extraction must stay deterministic, and a prompt
-- injected into it would make it something else -- but an exemption that lives
-- in an `if` is not reviewable, not versioned, and not attributable to anyone.
-- It is also invisible to every governance screen, which is how an exemption
-- quietly grows a second entry.
--
-- DELIBERATELY NO applies_to. That field is how guardrail.authorize selects the
-- policies it weighs, and this row is configuration the orchestrator reads by
-- name -- not an authority statement about who may do what. P7 drew exactly
-- this line and it is worth keeping: giving a configuration row an applies_to
-- turns it into an authority statement nobody wrote, and the vocabulary tests
-- would then (rightly) demand an effect and a role it has no business stating.
--
-- A MISSING ROW EXEMPTS NOTHING. If this migration is not applied, every
-- workflow -- extraction included -- must resolve its governance, and a
-- governance outage stops it. That is the intended direction of the failure,
-- but it means this row is not optional on a deployment that ingests documents.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'WorkflowGovernancePolicy',
       'governance',
       'Which workflows may run without a governance envelope. Everything not '
       'named here must resolve its governance before it runs; if it cannot be '
       'resolved, the workflow is blocked rather than run ungoverned.',
       jsonb_build_object(
         'policy_identifier', 'workflow_governance',
         'rules', jsonb_build_object(
             'ungoverned_workflows', jsonb_build_array('document_extraction')
         )
       ),
       '', 1, 1, now(), 'workflow_governance', now(), 'workflow_governance'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'workflow_governance'
       AND p.policy_status = 1
 );

COMMIT;
