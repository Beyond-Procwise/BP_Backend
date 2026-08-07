-- Guardrail enforcement layer: storage + policy content.
-- Spec: docs/superpowers/specs/2026-08-06-guardrail-enforcement-layer-design.md
BEGIN;

-- 1. Supplier clearance. Existing rows default to 'internal', which keeps
-- routine RFQ traffic flowing while blocking commercial-confidential and
-- personal content to suppliers nobody has cleared.
ALTER TABLE proc.bp_supplier
    ADD COLUMN IF NOT EXISTS clearance_level text NOT NULL DEFAULT 'internal';

-- 2. Direct subject -> role overrides. Permissions live in bp_policy; only
-- the mapping is tabular.
CREATE TABLE IF NOT EXISTS proc.bp_role_assignment (
    assignment_id bigserial PRIMARY KEY,
    subject       text NOT NULL,
    role          text NOT NULL,
    granted_by    text,
    granted_at    timestamptz NOT NULL DEFAULT now(),
    revoked_at    timestamptz
);

CREATE INDEX IF NOT EXISTS ix_bp_role_assignment_subject
    ON proc.bp_role_assignment (subject)
    WHERE revoked_at IS NULL;

-- 3. Policy content. Six rows, all active.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
VALUES
(
 'RoleDefinitionPolicy', 'security',
 'The four roles and the action classes each may perform.',
 '{
   "policy_identifier": "role_definition",
   "required_role": "Admin",
   "rules": {
     "roles": {
       "Viewer":   {"rank": 1, "allow": ["read"]},
       "Buyer":    {"rank": 2, "allow": ["read","compute","write"]},
       "Approver": {"rank": 3, "allow": ["read","compute","write","communicate","transact"]},
       "Admin":    {"rank": 4, "allow": ["read","compute","write","communicate","transact","share","configure","delegate"]}
     },
     "irreversible_classes": ["communicate","transact","share","configure","delegate"],
     "on_missing_role": "deny"
   }
 }'::jsonb,
 '', 1, 1, 'guardrail_migration', now()
),
(
 'RoleAssignmentPolicy', 'security',
 'Cognito group to role mapping and the no-principal default.',
 '{
   "policy_identifier": "role_assignment",
   "required_role": "Admin",
   "rules": {
     "claim": "cognito:groups",
     "group_to_role": {
       "bp-viewers": "Viewer",
       "bp-buyers": "Buyer",
       "bp-approvers": "Approver",
       "bp-admins": "Admin"
     },
     "no_principal_role": "Viewer",
     "unmapped_group_role": "Viewer",
     "multiple_groups": "highest_rank"
   }
 }'::jsonb,
 '', 1, 1, 'guardrail_migration', now()
),
(
 'EmailDispatchApprovalPolicy', 'email',
 'Outbound mail requires an approval verified against the store.',
 '{
   "policy_identifier": "email_dispatch_approval",
   "required_role": "Approver",
   "rules": {
     "approval_required": true,
     "verify_against": "proc.bp_approval",
     "accepted_status": ["approved"],
     "require_actioned_by": true,
     "trust_input_payload": false,
     "on_missing_approval": "deny"
   }
 }'::jsonb,
 'email_dispatch_agent', 1, 1, 'guardrail_migration', now()
),
(
 'EmailRecipientAllowlistPolicy', 'email',
 'Recipients must appear on the supplier master.',
 '{
   "policy_identifier": "email_recipient_allowlist",
   "required_role": "Approver",
   "rules": {
     "sources": ["proc.bp_supplier.contact_email_1","proc.bp_supplier.contact_email_2"],
     "match": "exact_casefold",
     "on_unknown_recipient": "deny_and_raise_review",
     "allow_recipients_from_email_body": false,
     "new_domain_requires_confirmation": true
   }
 }'::jsonb,
 'email_dispatch_agent, supplier_interaction_agent', 1, 1, 'guardrail_migration', now()
),
(
 'EmailSensitivityPolicy', 'email',
 'Outbound content class must not exceed the recipient supplier clearance.',
 '{
   "policy_identifier": "email_sensitivity",
   "required_role": "Admin",
   "rules": {
     "classes": ["public","internal","commercial_confidential","personal"],
     "order": {"public": 1, "internal": 2, "commercial_confidential": 3, "personal": 4},
     "default_supplier_clearance": "internal",
     "detectors": {
       "third_party_price":        {"enabled": true, "raises_to": "commercial_confidential"},
       "contract_prose":           {"enabled": true, "raises_to": "commercial_confidential"},
       "internal_staff_contact":   {"enabled": true, "raises_to": "personal"},
       "source_document_attached": {"enabled": true, "raises_to": "commercial_confidential"}
     },
     "rule": "content_class <= recipient_clearance",
     "on_undetermined_class": "deny",
     "on_missing_clearance": "use_default"
   }
 }'::jsonb,
 'email_dispatch_agent, email_drafting_agent', 1, 1, 'guardrail_migration', now()
),
(
 'EmailVolumePolicy', 'email',
 'Per-run and per-user outbound caps.',
 '{
   "policy_identifier": "email_volume",
   "required_role": "Admin",
   "rules": {
     "max_per_run": 20,
     "max_per_user_per_day": 50,
     "on_exceeded": "pause_and_raise_review"
   }
 }'::jsonb,
 'email_dispatch_agent', 1, 1, 'guardrail_migration', now()
);

COMMIT;
