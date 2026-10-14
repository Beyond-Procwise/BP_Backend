-- The abandoned-draft sweep: after this many quiet days a draft nobody sent and nobody abandoned is recorded as abandoned,
-- but only when the product tables confirm it was not sent. NOT applied anywhere. Governed: a missing or non-positive value
-- makes the sweep refuse to run.
BEGIN;

INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT 'EmailDraftSweepRules', 'email_sweep',
 'How long an email draft may sit with no send and no decision before it is recorded as abandoned.',
 $json${
  "policy_identifier": "email_draft_sweep_rules",
  "required_role": "Admin",
  "rules": {
    "abandon_after_days": 14,
    "batch_size": 500
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailDraftSweepRules');

COMMIT;
