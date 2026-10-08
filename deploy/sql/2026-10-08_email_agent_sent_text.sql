-- email_agent: the text that was SENT, and the diff from the model's draft; with a retention period.
--
-- NOT applied anywhere. bp_testdb first, bp_sqldb only after live verification (ruling 2026-10-08).
-- Reverses the earlier "store no sent text" ruling (decision 3, 2026-10-08): the learning job needs the
-- words, and capture without them was a gap, not a design choice.
--
-- RAW TEXT lives in ONE table, bp_draft_sent_text, so access to it can be granted separately from the derived
-- columns (scores, classes, changed figures) that stay in bp_draft_capture / bp_draft_outcome. Only the
-- email_agent_writer role is granted it (see the roles migration); the reader and PUBLIC have no access.
--
-- Two kinds of raw text exist: this table, and bp_draft_capture.draft_text (the model's own draft). Both age
-- out at the same configured period. Derived features are NOT deleted by the retention job.
--
-- Bank details are masked BEFORE storage (the application does this; "redactions" counts what was masked).

BEGIN;

CREATE TABLE IF NOT EXISTS email_agent.bp_draft_sent_text (
    outcome_id   BIGINT      PRIMARY KEY REFERENCES email_agent.bp_draft_outcome (outcome_id),
    capture_id   BIGINT      NOT NULL REFERENCES email_agent.bp_draft_capture (capture_id),
    sent_text    TEXT        NOT NULL,               -- plain text of what went out, bank details masked
    diff         JSONB,                              -- word ops that turn the draft into sent_text; NULL = no draft text left to compare
    text_hash    TEXT        NOT NULL,
    redactions   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    stored_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_draft_sent_text_stored ON email_agent.bp_draft_sent_text (stored_at);
REVOKE ALL ON email_agent.bp_draft_sent_text FROM PUBLIC;

COMMENT ON TABLE email_agent.bp_draft_sent_text IS
    'RAW TEXT. Restricted to email_agent_writer. Purged after the EmailTextRetention period.';
COMMENT ON COLUMN email_agent.bp_draft_sent_text.diff IS
    'JSON list of word ops: ["eq",n] keep n draft words, ["del","words"], ["ins","words"]. NULL when the draft text had already expired.';

-- When the model's own draft text was blanked by the retention job (draft_text becomes '' and this is set).
ALTER TABLE email_agent.bp_draft_capture ADD COLUMN IF NOT EXISTS text_expired_at TIMESTAMPTZ;
COMMENT ON COLUMN email_agent.bp_draft_capture.text_expired_at IS
    'Set when draft_text was blanked by retention. NULL = the text is still held. An outcome recorded after this has no edit distance.';

-- The retention period. Governed: a missing or non-positive value makes the job REFUSE and stores no raw text.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT 'EmailTextRetention', 'email_retention',
 'How long raw email text (sent text, diffs, and the model''s own drafts) is kept.',
 $json${
  "policy_identifier": "email_text_retention",
  "required_role": "Admin",
  "rules": {
    "raw_text_days": 90
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailTextRetention');

COMMIT;
