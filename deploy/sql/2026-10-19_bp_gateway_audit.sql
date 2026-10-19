BEGIN;
-- deploy/sql/2026-10-19_bp_gateway_audit.sql
--
-- The data gateway's audit trail: one row per gateway call, one row per gateway write. Approved 2026-10-19, to be applied to
-- bp_testdb first and promoted to bp_sqldb after review.
--
-- Both tables are append-only IN THE DATABASE, by the same mechanism and for the same reasons as proc.bp_agent_actions
-- (2026-09-16_bp_agent_actions_immutable.sql): the application role OWNS them, so REVOKE would be theatre; a BEFORE trigger
-- that raises is what binds an owner, and TRUNCATE needs its own statement-level trigger. INSERT stays open.
--
-- bp_gateway_call   : caller, agent, function, validated parameters, the record ids returned, latency (architecture: Audit).
-- bp_gateway_write  : actor, agent, inputs, provenance of the data relied on. Empty until the write functions exist (Step 9);
--                     (function, idempotency_key) is UNIQUE so a retried write cannot be recorded twice.
-- Idempotent. Reversible: 2026-10-19_bp_gateway_audit_rollback.sql.

CREATE TABLE IF NOT EXISTS proc.bp_gateway_call (
    call_id            text        PRIMARY KEY,
    ts                 timestamptz NOT NULL,
    caller_subject     text        NOT NULL,
    role               text,
    tenant_id          text,
    agent              text,
    request_id         text,
    function           text        NOT NULL,
    function_version   integer,
    params             jsonb       NOT NULL DEFAULT '{}'::jsonb,
    result_record_ids  jsonb       NOT NULL DEFAULT '[]'::jsonb,
    row_count          integer,
    truncated          boolean     NOT NULL DEFAULT false,
    ok                 boolean     NOT NULL,
    error_code         text,
    latency_ms         integer,
    scope_violation    boolean     NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_call_ts       ON proc.bp_gateway_call (ts DESC);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_call_function ON proc.bp_gateway_call (function, ts DESC);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_call_subject  ON proc.bp_gateway_call (caller_subject, ts DESC);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_call_request  ON proc.bp_gateway_call (request_id);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_call_refused  ON proc.bp_gateway_call (ts DESC) WHERE ok = false;

CREATE TABLE IF NOT EXISTS proc.bp_gateway_write (
    write_id         text        PRIMARY KEY,
    call_id          text,
    ts               timestamptz NOT NULL,
    actor            text        NOT NULL,
    agent            text,
    function         text        NOT NULL,
    idempotency_key  text        NOT NULL,
    inputs           jsonb       NOT NULL DEFAULT '{}'::jsonb,
    provenance       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    confirmation_id  text,
    result           jsonb,
    CONSTRAINT uq_bp_gateway_write_idem UNIQUE (function, idempotency_key)
);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_write_ts    ON proc.bp_gateway_write (ts DESC);
CREATE INDEX IF NOT EXISTS ix_bp_gateway_write_actor ON proc.bp_gateway_write (actor, ts DESC);

CREATE OR REPLACE FUNCTION proc.bp_gateway_audit_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only: % is refused. A gateway audit row records something that actually happened; it is never corrected '
        'and never tidied away. Write a new row instead.', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'integrity_constraint_violation';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_bp_gateway_call_append_only ON proc.bp_gateway_call;
CREATE TRIGGER tr_bp_gateway_call_append_only BEFORE UPDATE OR DELETE ON proc.bp_gateway_call
    FOR EACH ROW EXECUTE FUNCTION proc.bp_gateway_audit_append_only();
DROP TRIGGER IF EXISTS tr_bp_gateway_call_no_truncate ON proc.bp_gateway_call;
CREATE TRIGGER tr_bp_gateway_call_no_truncate BEFORE TRUNCATE ON proc.bp_gateway_call
    FOR EACH STATEMENT EXECUTE FUNCTION proc.bp_gateway_audit_append_only();

DROP TRIGGER IF EXISTS tr_bp_gateway_write_append_only ON proc.bp_gateway_write;
CREATE TRIGGER tr_bp_gateway_write_append_only BEFORE UPDATE OR DELETE ON proc.bp_gateway_write
    FOR EACH ROW EXECUTE FUNCTION proc.bp_gateway_audit_append_only();
DROP TRIGGER IF EXISTS tr_bp_gateway_write_no_truncate ON proc.bp_gateway_write;
CREATE TRIGGER tr_bp_gateway_write_no_truncate BEFORE TRUNCATE ON proc.bp_gateway_write
    FOR EACH STATEMENT EXECUTE FUNCTION proc.bp_gateway_audit_append_only();

COMMIT;
