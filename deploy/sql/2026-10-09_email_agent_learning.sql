-- Stage 6: what the platform learns from what reviewers do to a draft, and where each lesson goes.
--
-- Apply to bp_testdb first (non-prod). NOT applied to bp_sqldb. Reversible: _rollback.sql.
--
-- Every table here is a QUEUE or a CANDIDATE list. Nothing in this migration lets the job change a
-- prompt, a policy, a style profile, an exemplar set or a family config: a person decides each one.
-- Nothing holds the text that was sent; the only prose is the MODEL'S own draft (already in
-- bp_draft_capture) and the person's own instruction.
--
-- THE FACT RULE: a reviewer's replacement for a Postgres-backed figure goes to bp_dq_item ONLY, as an
-- unverified value for the data owner to check. It is never written to an eval candidate, an exemplar
-- candidate, a style rule or a classifier example.

BEGIN;

ALTER TABLE email_agent.bp_draft_outcome
    ADD COLUMN IF NOT EXISTS learning_processed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS learning_routes       JSONB;       -- e.g. ["fact","wording"]; '[]' = nothing to learn

CREATE INDEX IF NOT EXISTS ix_bp_draft_outcome_unlearned
    ON email_agent.bp_draft_outcome (outcome_id) WHERE outcome = 'sent' AND learning_processed_at IS NULL;

-- Route 1: a figure that came from Postgres was changed by a person.
CREATE TABLE IF NOT EXISTS email_agent.bp_dq_item (
    dq_id             BIGSERIAL PRIMARY KEY,
    outcome_id        BIGINT NOT NULL REFERENCES email_agent.bp_draft_outcome (outcome_id),
    capture_id        BIGINT NOT NULL,
    family_id         TEXT,
    fact_key          TEXT   NOT NULL,
    source            JSONB,                 -- the row it was read from: table, column, row_id, retrieved_at
    value_in_postgres TEXT,
    value_from_reviewer TEXT,                -- UNVERIFIED. For the data owner; never learned from.
    sent_by           TEXT,
    status            TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'dismissed')),
    note              TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_by       TEXT,
    resolved_at       TIMESTAMPTZ,
    UNIQUE (outcome_id, fact_key)
);

-- Route 2: a judgement of ours (price, deadline) was corrected. A candidate for the family's eval set.
CREATE TABLE IF NOT EXISTS email_agent.bp_eval_candidate (
    eval_id        BIGSERIAL PRIMARY KEY,
    outcome_id     BIGINT NOT NULL REFERENCES email_agent.bp_draft_outcome (outcome_id),
    capture_id     BIGINT NOT NULL,
    family_id      TEXT   NOT NULL,
    correction_key TEXT   NOT NULL,          -- which reasoned value, e.g. counter_price
    direction      TEXT   NOT NULL CHECK (direction IN ('raised', 'lowered', 'changed')),
    from_value     TEXT,
    to_value       TEXT,
    snapshot       JSONB,                    -- facts, reasoned, tone and brief at drafting time
    draft_text     TEXT,
    sent_by        TEXT,
    status         TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'exported', 'rejected')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (outcome_id, correction_key)
);

-- Routes 2 and 4: a pattern across reviewers that a person should look at. Never auto-applied.
CREATE TABLE IF NOT EXISTS email_agent.bp_review_item (
    review_id  BIGSERIAL PRIMARY KEY,
    family_id  TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('reasoning_guidance', 'wording_review')),
    signature  TEXT NOT NULL,
    evidence   JSONB,                        -- counts and outcome ids only
    status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'accepted', 'dismissed')),
    opened_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by TEXT,
    decided_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_review_item_open
    ON email_agent.bp_review_item (family_id, kind, signature) WHERE status = 'open';

-- Route 3: proposed style rules for ONE reviewer. Proposed until that person approves or edits them.
CREATE TABLE IF NOT EXISTS email_agent.bp_style_rule (
    rule_id      BIGSERIAL PRIMARY KEY,
    sent_by      TEXT NOT NULL,
    batch_id     TEXT NOT NULL,
    rule_key     TEXT NOT NULL,
    rule_text    TEXT NOT NULL,
    evidence     JSONB,                      -- the numbers the rule rests on
    status       TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'approved', 'edited', 'rejected', 'superseded')),
    edited_text  TEXT,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by   TEXT,
    decided_at   TIMESTAMPTZ,
    UNIQUE (sent_by, batch_id, rule_key)
);

-- Route 5: a reviewer's answer to "which kind of email is this?" is a labelled classifier example.
CREATE TABLE IF NOT EXISTS email_agent.bp_classifier_example (
    example_id       BIGSERIAL PRIMARY KEY,
    capture_id       BIGINT NOT NULL UNIQUE,
    request_text     TEXT NOT NULL,
    predicted_family TEXT,
    labeled_family   TEXT NOT NULL,
    labeled_by       TEXT,
    status           TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'exported', 'rejected')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Exemplars: drafts people barely touched. A candidate list; promotion is a person's decision.
CREATE TABLE IF NOT EXISTS email_agent.bp_exemplar_candidate (
    exemplar_id    BIGSERIAL PRIMARY KEY,
    capture_id     BIGINT NOT NULL UNIQUE,
    outcome_id     BIGINT NOT NULL,
    family_id      TEXT   NOT NULL,
    tone_variables JSONB,
    author         TEXT,                     -- sent_by
    reviewed_by    TEXT,
    edit_distance  NUMERIC(4,3),
    judge_overall  NUMERIC(3,2),
    draft_text     TEXT,                     -- the model's draft; the sent text was at most 15% different
    status         TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'approved', 'rejected', 'expired')),
    approved_by    TEXT,
    approved_at    TIMESTAMPTZ,
    review_after   DATE,                     -- approved_at + 12 months: re-reviewed, not trusted forever
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The thresholds. A missing value makes the job refuse to run; there is no default in the code.
INSERT INTO proc.bp_policy
    (policy_name, policy_type, policy_desc, policy_details,
     policy_linked_agents, policy_status, version, created_by, created_date)
SELECT 'EmailLearningRules', 'email_learning',
 'Thresholds for the job that learns from reviewers editing email drafts.',
 $json${
  "policy_identifier": "email_learning_rules",
  "required_role": "Admin",
  "rules": {
    "min_distinct_users": 3,
    "window_days": 90,
    "style_window_edits": 50,
    "min_edits_for_style_rule": 10,
    "max_style_rules": 15,
    "length_shorten_ratio": 0.85,
    "length_lengthen_ratio": 1.15,
    "high_divergence": 0.35,
    "wording_edit_min_words": 3,
    "exemplar_max_distance": 0.15,
    "exemplar_min_judge": 4,
    "exemplar_review_months": 12,
    "batch_size": 500
  }
}$json$::jsonb,
 '', 1, 1, 'email_assurance_migration', now()
WHERE NOT EXISTS (SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EmailLearningRules');

COMMIT;
