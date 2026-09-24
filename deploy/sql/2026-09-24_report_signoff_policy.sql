-- 2026-09-24  Report sign-off before release (RGA §8, D8): policy, not code.
-- ---------------------------------------------------------------------------
-- Ruled by the user 2026-09-24: every report needs sign-off before it leaves the
-- company, and that is a policy point, not hardcoded; Approvers (and Admins, who
-- outrank them) sign off.
--
-- ReportSignoffPolicy says WHICH reports need sign-off ("*" = all) and whether the
-- person who asked for a report may sign it off. Read by name (no applies_to), by
-- src/services/rga/signoff.py; an unreadable row holds every report.
--
-- ReportSignoffAuthorityPolicy is the stated permit for report.signoff (class
-- transact, irreversible, so without it the gate would refuse everyone).
--
-- Idempotent. Run against: bp_testdb, bp_sqldb.
-- ---------------------------------------------------------------------------
BEGIN;

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version, created_date, created_by,
    last_modified_date, last_modified_by)
SELECT 'ReportSignoffPolicy', 'approval',
       'Which reports need a person''s sign-off before they may leave the company, and '
       'whether the person who asked for a report may sign it off.',
       jsonb_build_object('policy_identifier', 'report_signoff',
         'rules', jsonb_build_object('requires_signoff', jsonb_build_array('*'),
                                     'self_approval', 'deny')),
       '', 1, 1, now(), 'report_signoff', now(), 'report_signoff'
 WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy p
                    WHERE p.policy_details->>'policy_identifier' = 'report_signoff'
                      AND p.policy_status = 1);

INSERT INTO proc.bp_policy (policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version, created_date, created_by,
    last_modified_date, last_modified_by)
SELECT 'ReportSignoffAuthorityPolicy', 'authority',
       'Who may sign off a report so it can leave the company.',
       jsonb_build_object('policy_identifier', 'report_signoff_authority',
         'applies_to', jsonb_build_array('report.signoff'),
         'required_role', 'Approver',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'report_signoff', now(), 'report_signoff'
 WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy p
                    WHERE p.policy_details->>'policy_identifier' = 'report_signoff_authority'
                      AND p.policy_status = 1);

COMMIT;
