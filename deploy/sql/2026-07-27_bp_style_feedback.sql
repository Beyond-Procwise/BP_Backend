BEGIN;
-- deploy/sql/2026-07-27_bp_style_feedback.sql
--
-- Phase 7. Did the draft survive contact with the person who sent it?
--
-- The measurement is the point, and so is what is NOT measured. These tables record how
-- FAR a sent email drifted from the draft, never what it said. There is no column here
-- that could hold correspondence, in any mode — the specification permits retaining the
-- sent text outside Modes A and C2, and this deliberately does not, because a
-- mode-dependent exception to "we do not keep your mail" is the one that gets forgotten.
-- A score is enough to know a profile is wrong; the text would only be enough to be
-- embarrassing.
--
-- Recompilation is never automatic. A sustained drift raises a SUGGESTION that a human
-- accepts or dismisses. Silently recompiling a profile would change how someone's mail
-- reads without them agreeing to it, which is the same failure invariant 6 exists to
-- prevent at approval time.

-- ===================== bp_style_divergence =====================
-- One row per draft that was later observed as sent.
CREATE TABLE IF NOT EXISTS proc.bp_style_divergence (
    divergence_id         BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    draft_id              BIGINT      NOT NULL REFERENCES proc.draft_rfq_emails(id) ON DELETE CASCADE,
    user_ref              TEXT        NOT NULL,
    intent                TEXT        REFERENCES proc.bp_style_intent(code),
    style_profile_id      BIGINT      REFERENCES proc.bp_style_profile(profile_id),
    style_profile_version INTEGER,

    -- 0.000 = sent exactly as drafted; 1.000 = nothing of the draft survived.
    -- Word-level edit distance, normalised by the longer of the two.
    score                 NUMERIC(4,3) NOT NULL CHECK (score >= 0 AND score <= 1),
    -- Supporting counts, so a score can be interpreted later without the text. A 0.4 over
    -- 20 words and a 0.4 over 300 words are not the same finding.
    drafted_words         INTEGER,
    sent_words            INTEGER,
    -- How the sent version was observed ('mailbox' | 'manual'). Retained because a score
    -- from a mailbox read and one a user typed in are different kinds of evidence.
    observed_via          TEXT        NOT NULL DEFAULT 'mailbox',
    observed_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One observation per draft. A draft cannot be sent twice, and a second row would
    -- double-count that draft in every average built from this table.
    UNIQUE (draft_id)
);
CREATE INDEX IF NOT EXISTS ix_bp_style_divergence_scope
    ON proc.bp_style_divergence (user_ref, intent, observed_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_style_divergence_profile
    ON proc.bp_style_divergence (style_profile_id);

-- ===================== bp_style_recompile_suggestion =====================
-- Raised when drift is sustained. Never acted on without a human.
CREATE TABLE IF NOT EXISTS proc.bp_style_recompile_suggestion (
    suggestion_id         BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_ref              TEXT        NOT NULL,
    intent                TEXT        NOT NULL REFERENCES proc.bp_style_intent(code),
    style_profile_id      BIGINT      REFERENCES proc.bp_style_profile(profile_id),
    style_profile_version INTEGER,

    -- The evidence, in the form a person can read: how many drafts, how far they drifted.
    observation_count     INTEGER     NOT NULL,
    mean_score            NUMERIC(4,3) NOT NULL,
    window_days           INTEGER     NOT NULL,
    reason                TEXT        NOT NULL,

    -- pending -> the user has not looked at it
    -- accepted -> they asked to recompile (which then follows the normal DRAFT + approve
    --             path; accepting a suggestion does NOT activate anything)
    -- dismissed -> they looked and disagreed; suppressed until fresh evidence accumulates
    status                TEXT        NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'accepted', 'dismissed')),
    actioned_by           TEXT,
    actioned_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT ck_bp_style_suggestion_actioned_has_actor
        CHECK (status = 'pending' OR (actioned_by IS NOT NULL AND actioned_at IS NOT NULL))
);
-- At most one open suggestion per scope. Without this, every sweep would raise another
-- one and the user would be nagged by a queue rather than told once.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_style_suggestion_open
    ON proc.bp_style_recompile_suggestion (user_ref, intent) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS ix_bp_style_suggestion_status
    ON proc.bp_style_recompile_suggestion (status, created_at DESC);

COMMIT;
