-- P7: the ungated doors get a rule, in the canonical action vocabulary.
--
-- This is not what P7 was originally written to be. The plan said "add
-- applies_to to the sixteen policies the gate cannot see", on the assumption
-- that they were rules waiting to be enforced. They are not: eleven of them are
-- scoring configuration, opportunity detectors and validation lists, read
-- directly by the code that needs them. Giving those an applies_to would have
-- made a supplier-ranking weights table into an authority statement.
--
-- The real gap was the opposite: actions with NO policy at all. Report export,
-- agent creation, policy reload, model training, workflow saving. Since the gate
-- learned its third answer they are UNRESOLVED rather than silently permitted --
-- correct, but a question asked forever is not governance either. These rows are
-- the answers.
--
-- Names come from services/actions.ACTIONS, which is closed and tested; roles
-- come from the RBAC set agreed on 2026-09-09. Every row states its effect,
-- because a permit is stated and never inferred.
--
-- INERT UNTIL WIRED: guardrail.authorize has four call sites and none of these
-- actions is among them. Adding a policy does not add a check. What it does is
-- mean that when the call site arrives (gap GA-3) the authority is already
-- decided, versioned and attributable, rather than being invented in the same
-- commit as the enforcement.
BEGIN;

-- #10 already holds the spend threshold DecisionEngine reads by name. It now
-- also states who may approve spend at all. required_role sits beside the rules,
-- so the threshold lookup is untouched.
UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(
               jsonb_set(policy_details, '{applies_to}', '["spend.approve"]'::jsonb, true),
               '{required_role}', '"Approver"'::jsonb, true
           ),
           '{rules,effect}', '"allow"'::jsonb, true
       ),
       version = version + 1,
       last_modified_by = 'action_authority',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'approval_threshold';

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT v.name, 'authority', v.descr,
       jsonb_build_object(
         'policy_identifier', v.slug,
         'required_role', v.role,
         'applies_to', v.actions,
         'rules', jsonb_build_object('effect', 'allow')
       ),
       '', 1, 1, now(), 'action_authority', now(), 'action_authority'
  FROM (VALUES
    ('ReportExportAuthorityPolicy',   'report_export_authority',
     'Who may take spend figures and supplier names out of the product.',
     'Admin',    '["report.export"]'::jsonb),
    ('AgentLifecycleAuthorityPolicy', 'agent_lifecycle_authority',
     'Who may create or delete an agent. A derived agent inherits reach, so this is a delegation.',
     'Admin',    '["agent.create","agent.delete"]'::jsonb),
    ('WorkflowAuthorityPolicy',       'workflow_authority',
     'Who may save or run a workflow graph.',
     'Admin',    '["workflow.save","workflow.run"]'::jsonb),
    ('GovernanceAuthorityPolicy',     'governance_authority',
     'Who may edit or reload the rules that govern everything else.',
     'Admin',    '["policy.write","policy.reload","prompt.write"]'::jsonb),
    ('ModelAuthorityPolicy',          'model_authority',
     'Who may retrain or repoint a model.',
     'Admin',    '["model.train"]'::jsonb),
    ('MailboxAuthorityPolicy',        'mailbox_authority',
     'Who may change which mailbox is watched.',
     'Admin',    '["mailbox.bind"]'::jsonb),
    ('SupplierClearanceAuthorityPolicy', 'supplier_clearance_authority',
     'Who may set a supplier clearance level. Needs a recorded contract or NDA reference.',
     'Admin',    '["supplier.clearance.set"]'::jsonb),
    ('DocumentIntakeAuthorityPolicy', 'document_intake_authority',
     'Who may put documents into the product and promote them to the record of truth.',
     'Buyer',    '["document.upload","document.promote"]'::jsonb),
    ('SupplierMasterAuthorityPolicy', 'supplier_master_authority',
     'Who may write to the supplier master.',
     'Buyer',    '["supplier.write"]'::jsonb),
    ('WebResearchAuthorityPolicy',    'web_research_authority',
     'Who may cause a query about a supplier to leave the tenant.',
     'Buyer',    '["research.web"]'::jsonb)
  ) AS v(name, slug, descr, role, actions)
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_name = v.name AND p.policy_status = 1
 );

COMMIT;
