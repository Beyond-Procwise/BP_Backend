-- D-60 / D-62 (backlog 1.23): who may answer a paused workflow's question is an administrator's setting.
-- FOR REVIEW. NOT APPLIED to any database. Idempotent: writes the starting row only if there is none,
-- so it never overwrites an administrator's choice. The same row is written by
-- POST /admin/permissions/seed (services.permission_settings.seed_defaults), so applying either is enough.
--
-- Values the Admin can choose (Admin Config -> Permissions): approver_only (default),
-- starter_or_approver, any_signed_in. If the row cannot be read the code answers Admin only.
-- Deliberately NO applies_to: the code reads this row by name; guardrail.authorize does not weigh it.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version, created_date, created_by
)
SELECT 'WorkflowQuestionAnswerPolicy', 'security',
       'When a workflow stops and asks a person something, who may answer. approver_only: only people with the approver permission, and never the person who started the run when the question is an approval. starter_or_approver: the person who started the run, or an approver. any_signed_in: anyone signed in.',
       jsonb_build_object('policy_identifier', 'workflow_question_answer',
                          'rules', jsonb_build_object('mode', 'approver_only')),
       '', 1, 1, now(), 'migration:2026-10-22_workflow_question_answer_policy'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'WorkflowQuestionAnswerPolicy' AND policy_status = 1
);

COMMIT;
