-- 2026-09-11 -- clear the stand-in names written where a person belongs.
--
-- Until today the code filled "who did this" columns with a name whenever it
-- was told nobody: settings.script_user ('AgentNick'), 'system', 'human',
-- 'api'. Those rows cannot be told apart from ones a person named AgentNick
-- created, and a stand-in can never match an approver, so the self-approval
-- bar could not fire on them. The code no longer writes them (e17aece,
-- 902480f and the standin-backfill commit). This clears the history to what
-- the new code would have written: NULL, nobody identified.
--
-- REVERSIBLE. Every value is copied to proc.bp_standin_actor_backfill before it
-- is cleared. To restore one table/column:
--
--   UPDATE proc.<table> t SET <column> = b.old_value
--     FROM proc.bp_standin_actor_backfill b
--    WHERE b.table_name = '<table>' AND b.column_name = '<column>'
--      AND t.<pk>::text = b.row_key;
--
-- IDEMPOTENT. A second run finds no stand-ins and writes nothing.
--
-- DELIBERATELY NOT TOUCHED:
--   bp_policy / bp_prompt created_by, last_modified_by = 'system'
--       NOT NULL, the tables' own default, meaning "shipped by a migration".
--   bp_summary.created_by = 'system'
--       NOT NULL; system-generated summaries, which is what it says.
--   bp_*_stg / bp_*_trgt created_by, last_modified_by = 'AgentNick'
--       The extraction agent really did create those rows, and they are the
--       financial record. Source data is not rewritten.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_standin_actor_backfill (
    backfill_id  BIGSERIAL PRIMARY KEY,
    table_name   TEXT NOT NULL,
    row_key      TEXT NOT NULL,
    column_name  TEXT NOT NULL,
    old_value    TEXT NOT NULL,
    cleared_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_standin_actor_backfill_table
    ON proc.bp_standin_actor_backfill (table_name, column_name);

DO $$
DECLARE
    standins CONSTANT text[] := ARRAY['AgentNick', 'system', 'human', 'api'];
    t record;
BEGIN
    FOR t IN SELECT * FROM (VALUES
        ('workflow_execution',        'execution_id',   'user_id'),
        ('routing',                   'process_id',     'created_by'),
        ('routing',                   'process_id',     'modified_by'),
        ('routing',                   'process_id',     'user_id'),
        ('routing',                   'process_id',     'user_name'),
        ('bp_requirement',            'requirement_id', 'created_by'),
        ('bp_workflow_input_request', 'request_id',     'answered_by'),
        ('bp_agent_group',            'group_id',       'created_by'),
        ('bp_agent_workflow',         'workflow_id',    'created_by')
    ) AS v(tbl, pk, col)
    LOOP
        EXECUTE format(
            'INSERT INTO proc.bp_standin_actor_backfill (table_name, row_key, column_name, old_value) '
            'SELECT %L, %I::text, %L, %I FROM proc.%I WHERE %I = ANY($1)',
            t.tbl, t.pk, t.col, t.col, t.tbl, t.col) USING standins;
        EXECUTE format('UPDATE proc.%I SET %I = NULL WHERE %I = ANY($1)',
                       t.tbl, t.col, t.col) USING standins;
    END LOOP;

    -- The run's triggered_by lives inside proc.routing.raw_data (json).
    INSERT INTO proc.bp_standin_actor_backfill (table_name, row_key, column_name, old_value)
    SELECT 'routing', process_id::text, 'raw_data.triggered_by', raw_data->>'triggered_by'
      FROM proc.routing
     WHERE raw_data->>'triggered_by' = ANY(standins);

    UPDATE proc.routing
       SET raw_data = jsonb_set(raw_data::jsonb, '{triggered_by}', 'null'::jsonb)::json
     WHERE raw_data->>'triggered_by' = ANY(standins);
END $$;

COMMIT;
