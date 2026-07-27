BEGIN;
-- deploy/sql/2026-07-27_bp_style_engine.sql
--
-- Email style-learning subsystem, Phase 0. See docs/style-engine/existing-email-inventory.md
-- for the reconciliation that produced this shape; it deviates from the original
-- specification in five ways, each recorded there and summarised here:
--
--   1. No tenant_id. This platform has no tenant dimension (src/api/auth.py). A column
--      that is always the same value buys nothing and implies isolation that does not exist.
--   2. Identity is `user_ref TEXT` — the Cognito `sub` — not a `persona_id`. "Persona"
--      already means "report voice" in proc.bp_prompt; two meanings would be misread.
--   3. Embeddings live in pgvector, on the row they describe — NOT in Qdrant, where every
--      other vector in this codebase lives. The reason is lifecycle, not preference. These
--      vectors are derived from correspondence, so they must die exactly when the row they
--      came from dies. As a column that is automatic and transactional. In Qdrant it is a
--      second delete against a second service, and a delete that silently fails leaves a
--      lossy copy of a customer's email sitting in a vector database after we told them it
--      was purged. For a subsystem whose whole premise is a provable data lifecycle, that
--      trade is worth one extra extension.
--   4. Drafts extend proc.draft_rfq_emails rather than landing in a new table. One draft
--      store means one provenance story, which is the whole point of the exercise.
--   5. BIGINT identity keys, not gen_random_uuid() — pgcrypto is not installed, and every
--      existing bp_ table keys this way.
--
-- Idempotent and transactional: safe to re-run. Reverse with the _rollback.sql of the
-- same name.

-- 1024 dimensions, matching BAAI/bge-large-en-v1.5 (config/settings.py: embedding_model,
-- vector_size). Reusing the platform's existing embedding model rather than introducing a
-- second one — a profile compiled against one model and retrieved against another would
-- silently return nonsense.
CREATE EXTENSION IF NOT EXISTS vector;

-- ===================== bp_style_intent =====================
-- Controlled vocabulary for "what kind of email is this". Merges the codes the
-- specification asked for with the ones EmailDraftingAgent._normalise_interaction_type
-- already emits in production, so there is one vocabulary rather than two.
--
-- `_all` is a scope, not a communication type. It exists so a user-level profile can
-- carry a real foreign key and participate in the one-active-profile-per-scope unique
-- index. A NULL intent could not: Postgres treats NULLs as distinct, so a partial unique
-- index over a nullable column would happily allow two active user-level profiles.
CREATE TABLE IF NOT EXISTS proc.bp_style_intent (
    code        TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    description TEXT,
    -- The legacy interaction_type this code supersedes, where one exists. Lets the
    -- existing normaliser be repointed at this table without a lookup dictionary.
    legacy_code TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO proc.bp_style_intent (code, label, description, legacy_code) VALUES
    ('_all',                  'Any intent',            'User-level profile covering every communication type. The default scope.', NULL),
    ('rfq_invite',            'RFQ invitation',        'Inviting a supplier to quote.',                                  'rfq'),
    ('clarification_request', 'Clarification request', 'Asking a supplier to explain or complete a response.',           'clarification'),
    ('negotiation_counter',   'Negotiation counter',   'Countering a supplier position on price or terms.',              'negotiation'),
    ('award_notification',    'Award notification',    'Telling a supplier they have won.',                              'award'),
    ('supplier_rejection',    'Supplier rejection',    'Telling a supplier they have not won.',                          NULL),
    ('escalation',            'Escalation',            'Raising an unresolved issue to a higher authority.',             NULL),
    ('contract_variation',    'Contract variation',    'Proposing or confirming a change to agreed terms.',              NULL),
    ('exit_notification',     'Exit notification',     'Ending a supplier relationship or contract.',                    NULL),
    ('internal_update',       'Internal update',       'Status update to colleagues rather than suppliers.',             'update'),
    ('follow_up',             'Follow-up',             'Chasing an outstanding response.',                               'follow_up'),
    ('reminder',              'Reminder',              'Restating an approaching deadline.',                             'reminder'),
    ('thank_you',             'Thank you',             'Acknowledging a response or a completed piece of work.',         'thank_you')
ON CONFLICT (code) DO NOTHING;

-- ===================== bp_style_ingest_staging =====================
-- A queue, not a store. Rows are hard-deleted when the batch compiles, or by the TTL
-- sweep, whichever comes first. Nothing downstream may read from here after compilation:
-- the profile is the artifact, and the emails that produced it do not survive it.
CREATE TABLE IF NOT EXISTS proc.bp_style_ingest_staging (
    ingest_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_ref     TEXT        NOT NULL,
    batch_id     TEXT        NOT NULL,
    intent       TEXT        REFERENCES proc.bp_style_intent(code),
    subject      TEXT,
    body         TEXT        NOT NULL,
    source_ref   TEXT,
    submitted_by TEXT        NOT NULL,
    -- Embedded at compile time so the batch can be clustered, intent-classified and
    -- checked for near-duplicates before a profile is written. Deliberately a column on
    -- the staging row rather than a separate store: when the batch is purged the vector
    -- goes with it in the same DELETE. A vector that outlived its email would be a lossy
    -- copy of correspondence we promised not to keep.
    embedding    vector(1024),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    purge_after  TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours')
);
CREATE INDEX IF NOT EXISTS ix_bp_style_ingest_staging_purge ON proc.bp_style_ingest_staging (purge_after);
CREATE INDEX IF NOT EXISTS ix_bp_style_ingest_staging_batch ON proc.bp_style_ingest_staging (batch_id);
CREATE INDEX IF NOT EXISTS ix_bp_style_ingest_staging_user  ON proc.bp_style_ingest_staging (user_ref);

-- ===================== bp_style_profile =====================
-- The artifact. A versioned description of writing HABITS — structure, register,
-- lexicon, behaviour. It holds no correspondence, and the no-content invariant is
-- tested against source n-grams rather than trusted.
--
-- Immutable once approved: recompilation inserts a new version and lands it in DRAFT.
-- Nothing auto-activates.
CREATE TABLE IF NOT EXISTS proc.bp_style_profile (
    profile_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_ref        TEXT        NOT NULL,
    intent          TEXT        NOT NULL REFERENCES proc.bp_style_intent(code),
    version         INTEGER     NOT NULL,
    state           TEXT        NOT NULL CHECK (state IN ('UNCOMPILED','DRAFT','APPROVED','SUPERSEDED')),
    profile_json    JSONB       NOT NULL,
    exemplar_count  INTEGER     NOT NULL,
    -- Which staging batch produced this, for audit. The batch itself is gone by now.
    source_batch_id TEXT,
    compiled_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_by     TEXT,
    approved_at     TIMESTAMPTZ,
    is_active       BOOLEAN     NOT NULL DEFAULT FALSE,
    UNIQUE (user_ref, intent, version),
    -- Only an APPROVED profile may be active, and an approved row must say who and when.
    CONSTRAINT ck_bp_style_profile_active_is_approved
        CHECK (NOT is_active OR state = 'APPROVED'),
    CONSTRAINT ck_bp_style_profile_approved_has_approver
        CHECK (state <> 'APPROVED' OR (approved_by IS NOT NULL AND approved_at IS NOT NULL))
);
-- At most one active profile per scope. This is what makes Phase 1's "approving n+1
-- deactivates n" atomic rather than aspirational: a transaction that fails to stand
-- the old row down cannot commit.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_style_profile_active
    ON proc.bp_style_profile (user_ref, intent) WHERE is_active;
CREATE INDEX IF NOT EXISTS ix_bp_style_profile_lookup ON proc.bp_style_profile (user_ref, intent, version DESC);
CREATE INDEX IF NOT EXISTS ix_bp_style_profile_state  ON proc.bp_style_profile (state);

-- ===================== bp_style_exemplar =====================
-- Two or three emails shown to the model to illustrate what compliance with the
-- profile looks like. Either platform-generated fiction ('synthetic') or read from a
-- mailbox the customer controls ('customer_retained', Mode C only).
--
-- Selection is by (user_ref, intent, is_active), then ordered by proximity to the task
-- at hand. Under Mode A/B that ordering is close to a formality — there are only a few
-- rows. It earns its keep under Mode C, where a bound mailbox can offer dozens of
-- candidates and three have to be chosen, and in Phase 7, where drift is measured as
-- distance between what we drafted and what was actually sent.
CREATE TABLE IF NOT EXISTS proc.bp_style_exemplar (
    exemplar_id         BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_ref            TEXT        NOT NULL,
    intent              TEXT        NOT NULL REFERENCES proc.bp_style_intent(code),
    origin              TEXT        NOT NULL CHECK (origin IN ('synthetic','customer_retained')),
    -- The profile version this set was generated to demonstrate. Recompiling stands the
    -- previous set down rather than mixing generations.
    profile_version_ref INTEGER     NOT NULL,
    subject             TEXT,
    body                TEXT        NOT NULL,
    embedding           vector(1024),
    -- Which model produced the vector. A profile embedded with one model and retrieved
    -- with another returns confident nonsense, and the failure is silent, so the model is
    -- recorded rather than assumed.
    embedding_model     TEXT,
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_bp_style_exemplar_lookup
    ON proc.bp_style_exemplar (user_ref, intent) WHERE is_active;
-- Cosine, matching how the platform's BGE vectors are normalised and compared elsewhere.
CREATE INDEX IF NOT EXISTS ix_bp_style_exemplar_embedding
    ON proc.bp_style_exemplar USING hnsw (embedding vector_cosine_ops);

-- ===================== bp_mailbox_binding =====================
-- Phase 5 (Mode C). Created now so the schema is coherent and the draft provenance
-- foreign key resolves; expected to stay empty until mailbox binding is built.
--
-- credential_ref is constrained to a Secrets Manager ARN at the database level, so
-- invariant 8 ("never a token or password") is structurally impossible to violate here
-- rather than merely tested. Note the pre-existing IMAP credentials in .env remain a
-- known gap outside this table — see E4(b) in the inventory.
CREATE TABLE IF NOT EXISTS proc.bp_mailbox_binding (
    binding_id         BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_ref           TEXT        NOT NULL,
    provider           TEXT        NOT NULL CHECK (provider IN ('graph','gmail','imap')),
    mailbox_address    TEXT        NOT NULL,
    role               TEXT        NOT NULL CHECK (role IN ('exemplar_source','draft_target','both')),
    credential_ref     TEXT        NOT NULL
        CONSTRAINT ck_bp_mailbox_binding_credential_is_arn
        CHECK (credential_ref LIKE 'arn:aws:secretsmanager:%'),
    scope_policy_ref   TEXT,
    -- Evidence that a read against a control mailbox was DENIED. A binding may not be
    -- activated without it.
    scope_verified_at  TIMESTAMPTZ,
    scope_evidence_ref TEXT,
    last_health_check  TIMESTAMPTZ,
    health_state       TEXT        NOT NULL DEFAULT 'OK' CHECK (health_state IN ('OK','DEGRADED','REVOKED')),
    is_active          BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_bp_mailbox_binding_active_needs_scope_proof
        CHECK (NOT is_active OR scope_verified_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS ix_bp_mailbox_binding_user ON proc.bp_mailbox_binding (user_ref) WHERE is_active;

-- ===================== draft provenance =====================
-- proc.draft_rfq_emails is the existing, live draft store (18 rows). Extending it keeps
-- one drafting path and one provenance story. Every column is nullable and style_-prefixed:
-- drafts produced before this subsystem, and drafts typed by hand through
-- POST /workflows/email/prepare, legitimately have no style provenance and must not be
-- made to look as though they do.
ALTER TABLE proc.draft_rfq_emails
    ADD COLUMN IF NOT EXISTS style_user_ref            TEXT,
    ADD COLUMN IF NOT EXISTS style_intent              TEXT REFERENCES proc.bp_style_intent(code),
    ADD COLUMN IF NOT EXISTS style_mode                TEXT CHECK (style_mode IN ('A','B','C1','C2')),
    ADD COLUMN IF NOT EXISTS style_profile_id          BIGINT REFERENCES proc.bp_style_profile(profile_id),
    ADD COLUMN IF NOT EXISTS style_profile_version     INTEGER,
    -- 0 profile hit | 1 user-level fallback | 2 tenant default | 3 platform baseline or
    -- source unreachable. NULL means the draft was not generated by this subsystem at all.
    ADD COLUMN IF NOT EXISTS style_fallback_level      SMALLINT CHECK (style_fallback_level BETWEEN 0 AND 3),
    ADD COLUMN IF NOT EXISTS style_exemplar_ids        BIGINT[],
    ADD COLUMN IF NOT EXISTS style_exemplar_set_hash   TEXT,
    ADD COLUMN IF NOT EXISTS style_mailbox_binding_id  BIGINT REFERENCES proc.bp_mailbox_binding(binding_id),
    ADD COLUMN IF NOT EXISTS style_message_ids         TEXT[],
    ADD COLUMN IF NOT EXISTS style_retrieved_at        TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS style_model_id            TEXT,
    -- The proc.bp_prompt.version of the system prompt that produced this draft. Reuses
    -- the existing governance registry rather than forking a second versioning scheme.
    ADD COLUMN IF NOT EXISTS style_prompt_version      INTEGER,
    -- Identifier returned by the mail provider when the draft is written back (Phase 6).
    ADD COLUMN IF NOT EXISTS external_draft_ref        TEXT;

CREATE INDEX IF NOT EXISTS ix_draft_rfq_emails_style_profile
    ON proc.draft_rfq_emails (style_profile_id) WHERE style_profile_id IS NOT NULL;

-- ===================== configuration =====================
-- Settings live in the existing key/JSONB config store, not a new table and not .env,
-- so deployment mode can be changed without a redeploy. There is no tenant dimension,
-- so this is a single deployment-wide row.
INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by)
VALUES ('style_engine', '{
    "deployment_mode": "A",
    "min_exemplars": 3,
    "staging_ttl_hours": 24
  }'::jsonb, 'seed')
ON CONFLICT (config_key) DO NOTHING;

COMMIT;
