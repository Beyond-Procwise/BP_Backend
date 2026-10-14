BEGIN;
DROP TABLE IF EXISTS email_agent.bp_exemplar_candidate, email_agent.bp_classifier_example, email_agent.bp_style_rule,
                     email_agent.bp_review_item, email_agent.bp_eval_candidate, email_agent.bp_dq_item;
DROP INDEX IF EXISTS email_agent.ix_bp_draft_outcome_unlearned;
ALTER TABLE email_agent.bp_draft_outcome DROP COLUMN IF EXISTS learning_processed_at, DROP COLUMN IF EXISTS learning_routes;
DELETE FROM proc.bp_policy WHERE policy_name = 'EmailLearningRules' AND created_by = 'email_assurance_migration';
COMMIT;
