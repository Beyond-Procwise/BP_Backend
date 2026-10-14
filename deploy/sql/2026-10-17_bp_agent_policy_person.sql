-- Agent policy governance, stage 5b: a sign-in id resolves to a name wherever it is shown.
-- One row per person who has made an agent-policy write through the gateway: the sign-in id
-- (Cognito subject) and the email their sign-in carried, so "Approved by" and an approval's
-- history can show who by name. Nothing else records the email of whoever decided an approval or
-- saved a version. Written by services.agent_policy.people.remember; a sign-in with no email
-- writes nothing and never erases a stored one. Additive and idempotent; applied to bp_testdb and
-- bp_sqldb.
BEGIN;
CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_person (
    subject        TEXT PRIMARY KEY,
    email          TEXT NOT NULL,
    first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMIT;
