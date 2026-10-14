-- Shadow mode: the observation record, and the enrolment list that ships EMPTY.
--
-- Sixteen of the nineteen active policies carry no applies_to, so the gate
-- cannot see them. Giving them one turns dormant rules into refusals, and
-- nobody knows what would be refused or to whom. This is what makes that a
-- measurement rather than a guess.
--
-- shadow_actions ships empty ON PURPOSE, the same way EmailReplyAutonomyPolicy's
-- auto_reply_intents does: nothing is shadowed on day one, and enrolling an
-- action is a governed edit rather than a deploy. Every entry MUST carry an
-- "until" -- an enrolment without an expiry is not honoured, so shadow mode
-- cannot become the permanent state by nobody getting round to it.
--
-- email.send and approval.email can never be enrolled. That is enforced in
-- services/guardrail.NEVER_SHADOW, not here, because a list in a row can be
-- edited and the point of those two is that they cannot.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_policy_observation (
    observation_id    BIGSERIAL PRIMARY KEY,
    observed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    action            TEXT        NOT NULL,
    action_class      TEXT,
    principal_subject TEXT,
    role              TEXT,
    verdict           TEXT        NOT NULL,
    would_have_denied BOOLEAN     NOT NULL,
    shadowed          BOOLEAN     NOT NULL DEFAULT false,
    policy_id         TEXT,
    policy_name       TEXT,
    policy_version    INTEGER,
    reason            TEXT,
    evidence          JSONB
);

CREATE INDEX IF NOT EXISTS ix_bp_policy_observation_observed
    ON proc.bp_policy_observation (observed_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_policy_observation_action
    ON proc.bp_policy_observation (action, observed_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_policy_observation_shadowed
    ON proc.bp_policy_observation (shadowed) WHERE shadowed;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT
    'ShadowModePolicy',
    'security',
    'Which actions are observed rather than enforced, and until when. Empty means nothing is shadowed.',
    '{
       "policy_identifier": "shadow_mode",
       "required_role": "Admin",
       "rules": {
         "shadow_actions": [],
         "note": "Each entry is {\"action\": \"<name>\", \"until\": \"<ISO-8601>\"}. An entry without until is ignored. email.send and approval.email can never be enrolled."
       }
     }'::jsonb,
    '', 1, 1, now(), 'shadow_mode_migration', now(), 'shadow_mode_migration'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
     WHERE policy_name = 'ShadowModePolicy' AND policy_status = 1
);

COMMIT;
