-- 2026-10-08  Agent policy governance, stage 1.
-- Agent policies are NOT proc.bp_policy rows (ruling B, 2026-10-08): that table holds
-- role permissions and tuning settings and is left untouched. Additive, idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_business_area (
    area_name        TEXT PRIMARY KEY,
    id_prefix        TEXT NOT NULL UNIQUE CHECK (id_prefix ~ '^[A-Z]{3}$'),
    sub_areas        TEXT[] NOT NULL DEFAULT ARRAY['General']
        CHECK ('General' = ANY (sub_areas)),
    -- Highest number ever issued under this prefix. Only ever increases, so an id is never reused.
    last_number      INTEGER NOT NULL DEFAULT 0 CHECK (last_number >= 0),
    never_suggest    BOOLEAN NOT NULL DEFAULT FALSE,
    second_reviewer  BOOLEAN NOT NULL DEFAULT FALSE,
    is_unassigned    BOOLEAN NOT NULL DEFAULT FALSE,
    last_modified_by TEXT NOT NULL DEFAULT 'seed',
    last_modified_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE proc.bp_business_area IS
    'Where an agent policy comes from (as its document states), never what it applies to. Admin-editable.';

INSERT INTO proc.bp_business_area (area_name, id_prefix, sub_areas, never_suggest, second_reviewer, is_unassigned) VALUES
  ('Unassigned',            'GEN', ARRAY['General'], FALSE, FALSE, TRUE),
  ('Finance',               'FIN', ARRAY['General','Refunds and credits','Payments','Expenses'], FALSE, TRUE, FALSE),
  ('Procurement',           'PRC', ARRAY['General','Sourcing','Purchase orders','Suppliers'], FALSE, FALSE, FALSE),
  ('Customer operations',   'CUS', ARRAY['General','Refunds','Complaints'], FALSE, FALSE, FALSE),
  ('Legal and compliance',  'LEG', ARRAY['General','Contracts','Data protection'], TRUE, FALSE, FALSE),
  ('Security',              'SEC', ARRAY['General','Access','Data handling'], TRUE, FALSE, FALSE),
  ('People',                'PPL', ARRAY['General'], FALSE, FALSE, FALSE),
  ('Operations',            'OPS', ARRAY['General'], FALSE, FALSE, FALSE)
ON CONFLICT (area_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS proc.bp_orchestrator_registry (
    registry_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('checkpoint','action','input')),
    name         TEXT NOT NULL,
    -- For 'action' and 'input': the checkpoint they belong to. NULL for a checkpoint.
    checkpoint   TEXT,
    plain        TEXT NOT NULL,
    value_type   TEXT CHECK (value_type IN ('string','number','boolean','date','list')),
    -- Where an input comes from. This release registers 'action' only (ruling: lookups and
    -- running totals belong to the orchestrator team and are not registered yet).
    source       TEXT CHECK (source IS NULL OR source = 'action' OR source LIKE 'lookup:%' OR source LIKE 'total:%'),
    -- 'live' = the orchestrator supplies it today. 'planned' = named but not yet supplied;
    -- a policy needing it shows "Can't be enforced yet".
    status       TEXT NOT NULL DEFAULT 'live' CHECK (status IN ('live','planned')),
    seeded_from  TEXT NOT NULL DEFAULT 'manual',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_orch_registry_checkpoint CHECK ((kind = 'checkpoint') = (checkpoint IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_orchestrator_registry_key
    ON proc.bp_orchestrator_registry (kind, name, COALESCE(checkpoint, ''));

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy (
    policy_key      TEXT PRIMARY KEY CHECK (policy_key ~ '^[A-Z]{3}-[0-9]{4,}$'),
    area_name       TEXT REFERENCES proc.bp_business_area (area_name),
    status          TEXT NOT NULL CHECK (status IN ('draft','live','retired')),
    live_version    INTEGER,
    latest_version  INTEGER NOT NULL,
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_bp_agent_policy_live CHECK ((status = 'live') = (live_version IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS ix_bp_agent_policy_status ON proc.bp_agent_policy (status);

CREATE TABLE IF NOT EXISTS proc.bp_agent_policy_version (
    policy_key     TEXT NOT NULL REFERENCES proc.bp_agent_policy (policy_key),
    version        INTEGER NOT NULL CHECK (version >= 1),
    saved_as       TEXT NOT NULL CHECK (saved_as IN ('draft','live','retired')),
    form_state     JSONB NOT NULL,
    compiled       JSONB,
    problems       JSONB NOT NULL DEFAULT '[]',
    confidence     JSONB,
    change_note    TEXT,
    saved_by       TEXT NOT NULL,
    saved_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (policy_key, version)
);

CREATE OR REPLACE FUNCTION proc.bp_agent_policy_version_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'proc.bp_agent_policy_version rows cannot be changed once saved (%)', TG_OP;
END $$;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_version_immutable ON proc.bp_agent_policy_version;
CREATE TRIGGER tr_bp_agent_policy_version_immutable
    BEFORE UPDATE OR DELETE ON proc.bp_agent_policy_version
    FOR EACH ROW EXECUTE FUNCTION proc.bp_agent_policy_version_immutable();

-- An id, once issued, is never freed: policies are retired, not deleted.
CREATE OR REPLACE FUNCTION proc.bp_agent_policy_no_delete() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'agent policies are retired, never deleted';
END $$;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_no_delete ON proc.bp_agent_policy;
CREATE TRIGGER tr_bp_agent_policy_no_delete
    BEFORE DELETE ON proc.bp_agent_policy
    FOR EACH ROW EXECUTE FUNCTION proc.bp_agent_policy_no_delete();

INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by) VALUES
  ('agent_policy_settings', '{
     "response_time": "PT4H",
     "response_time_basis": "clock",
     "on_missing_data": {"approve": "fail_closed", "block": "fail_closed", "notify": "fail_closed"},
     "live_conflict_repeat": 5,
     "learning": {"min_decisions": 30, "min_days": 30, "min_approvers": 3, "wilson_lower": 0.85,
                  "median_seconds_floor": 30, "not_yet_more": 30, "dismiss_more": 30}
   }'::jsonb, 'seed')
ON CONFLICT (config_key) DO NOTHING;

COMMIT;
