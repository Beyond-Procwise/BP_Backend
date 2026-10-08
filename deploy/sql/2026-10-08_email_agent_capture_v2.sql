-- email_agent capture v2: the outputs of Stages 1-4, accountability, and readiness.
--
-- Apply to bp_testdb first (non-prod). NOT applied to bp_sqldb. Reversible: _rollback.sql.
--
-- CONVENTION, applied to every new column below and relied on by the learning job:
--   SQL NULL                 = NOT CAPTURED (the stage did not run, or the row predates it)
--   '[]' / '{}' / 'none'     = CAPTURED, and nothing applied
--   stage_status.<stage>     = captured | empty | not_run | unavailable (+ reason)
-- so "no exemplars were found" and "exemplar retrieval never ran" can be told apart later.
--
-- user_id is dropped: it conflated the agent that started a draft with the person who owns it.
-- It is replaced by initiated_by (+ kind). The reviewer and sender are recorded on the OUTCOME,
-- because they are known only when the draft is sent. The table held no rows when this was written.

BEGIN;

ALTER TABLE email_agent.bp_draft_capture
    DROP COLUMN IF EXISTS user_id,
    ADD COLUMN IF NOT EXISTS initiated_by          TEXT,
    ADD COLUMN IF NOT EXISTS initiated_by_kind     TEXT CHECK (initiated_by_kind IN ('agent', 'user')),
    ADD COLUMN IF NOT EXISTS family_source         TEXT CHECK (family_source IN ('declared', 'classified', 'fallback')),
    ADD COLUMN IF NOT EXISTS classification        JSONB,   -- confidence, candidates, rejected lookup keys
    ADD COLUMN IF NOT EXISTS clarification         JSONB,   -- {question, options, resolution}; '{}' = none needed
    ADD COLUMN IF NOT EXISTS lookup_keys           JSONB,   -- candidates only, confirmed against Postgres elsewhere
    ADD COLUMN IF NOT EXISTS user_instruction      TEXT,
    ADD COLUMN IF NOT EXISTS tone_variables        JSONB,
    ADD COLUMN IF NOT EXISTS tone_sources          JSONB,   -- per variable: postgres | user_instruction | default
    ADD COLUMN IF NOT EXISTS exemplar_ids          JSONB,   -- '[]' when retrieval ran and found none
    ADD COLUMN IF NOT EXISTS exemplar_scope        TEXT,    -- user | organisation | none
    ADD COLUMN IF NOT EXISTS brief                 JSONB,
    ADD COLUMN IF NOT EXISTS assumption_items      JSONB,   -- [{id, key, text, resolution}]
    ADD COLUMN IF NOT EXISTS assumptions_resolution JSONB,  -- {id: {action, value, by, at}}
    ADD COLUMN IF NOT EXISTS judge                 JSONB,   -- {status, scores, overall} | {status, reason}
    ADD COLUMN IF NOT EXISTS authority             JSONB,   -- {verdict: within|exceeds|unresolved, ...}
    ADD COLUMN IF NOT EXISTS stage_status          JSONB,
    ADD COLUMN IF NOT EXISTS ready                 BOOLEAN,
    ADD COLUMN IF NOT EXISTS needs_redraft         BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS ready_at              TIMESTAMPTZ;

ALTER TABLE email_agent.bp_draft_outcome
    ADD COLUMN IF NOT EXISTS reviewed_by    TEXT,   -- the human who approved it (bp_approval.actioned_by)
    ADD COLUMN IF NOT EXISTS sent_by        TEXT,   -- the signed-in principal at send time
    ADD COLUMN IF NOT EXISTS abandoned_by   TEXT,
    ADD COLUMN IF NOT EXISTS abandon_reason TEXT;

-- OUT OF SCOPE until inbound email integration: nothing populates these three columns today,
-- and nothing should be read into their being empty. They are NULL = "not captured", never
-- "the supplier did not reply".
COMMENT ON COLUMN email_agent.bp_draft_outcome.supplier_replied IS
    'OUT OF SCOPE until inbound email integration. NULL means not captured, not "no reply".';
COMMENT ON COLUMN email_agent.bp_draft_outcome.reply_latency_s IS
    'OUT OF SCOPE until inbound email integration. NULL means not captured.';
COMMENT ON COLUMN email_agent.bp_draft_outcome.issue_resolved IS
    'OUT OF SCOPE until inbound email integration. NULL means not captured.';

COMMIT;
