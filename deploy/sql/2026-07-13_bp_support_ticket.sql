-- proc.bp_support_ticket — what the user asked support, and what happened next.
--
-- The ticket is written BEFORE the notification email is attempted. If SES is down,
-- the user's problem must not vanish with it: a support request that exists only in
-- an email that failed to send is a support request nobody will ever see.
-- email_status records whether the admin was actually told.

CREATE TABLE IF NOT EXISTS proc.bp_support_ticket (
    ticket_id       BIGSERIAL PRIMARY KEY,
    reference       TEXT UNIQUE NOT NULL,          -- human-quotable, e.g. SUP-4F2A91

    user_name       TEXT,
    user_email      TEXT,
    session_id      TEXT,

    -- What they said, and what the agent made of it.
    message         TEXT NOT NULL,
    category        TEXT,                          -- e.g. upload | extraction | access | data | other
    -- 'resolved'  -> the agent answered it; no human needed
    -- 'escalated' -> the agent could not resolve it; admin notified
    outcome         TEXT NOT NULL,
    agent_reply     TEXT,

    -- Did the admin actually get told? 'sent' | 'failed' | 'not_required'
    email_status    TEXT NOT NULL DEFAULT 'not_required',
    email_error     TEXT,
    notified_admin  TEXT,

    status          TEXT NOT NULL DEFAULT 'open',  -- open | in_progress | closed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_bp_support_ticket_status  ON proc.bp_support_ticket (status);
CREATE INDEX IF NOT EXISTS ix_bp_support_ticket_outcome ON proc.bp_support_ticket (outcome);
CREATE INDEX IF NOT EXISTS ix_bp_support_ticket_created ON proc.bp_support_ticket (created_at DESC);
