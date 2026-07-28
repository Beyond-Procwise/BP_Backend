-- Resync every identity/serial sequence in the proc schema that has fallen
-- behind its own data. Supersedes 2026-07-28_resync_governance_identity.sql,
-- which fixed two of these tables by name before the rest were found.
--
-- Symptom, from proc.bp_prompt:
--
--   duplicate key value violates unique constraint "bp_prompt_pkey"
--   DETAIL: Key (prompt_id)=(4) already exists.
--
-- The sequence was handing out 5 while the table already held ids up to 100, so
-- every application INSERT collided on the primary key. That took out POST
-- /agents (creating an agent from the workspace), the extraction-feedback hint
-- writer, and any policy insert — and all three reported it to the user as
-- "I couldn't retrieve that", because the safety gate strips the constraint name
-- before it reaches the browser. It reads like a failed lookup, not a failed write.
--
-- A survey of all 51 sequences in the schema found NINE behind their data,
-- including proc.bp_fx_rates (which throws on every startup as the rate loader
-- tries to insert) and the six raw ingestion tables, whose sequences sat at 1
-- against ids up to 193,568. Those survive only because the loader writes their
-- ids explicitly; the first insert that does not would fail on row one.
--
-- Rather than name today's nine, this repairs whatever is behind at the time it
-- runs. Idempotent, and a no-op when every sequence is already ahead. Safe to
-- re-run after any restore or bulk load — both of which are how a sequence gets
-- left behind in the first place.

DO $$
DECLARE
    r        RECORD;
    seq_name TEXT;
    max_id   BIGINT;
    last_val BIGINT;
    called   BOOLEAN;
    next_val BIGINT;
    fixed    INT := 0;
BEGIN
    FOR r IN
        SELECT c.table_name, c.column_name
        FROM information_schema.columns c
        WHERE c.table_schema = 'proc'
          AND pg_get_serial_sequence('proc.' || quote_ident(c.table_name), c.column_name) IS NOT NULL
    LOOP
        seq_name := pg_get_serial_sequence('proc.' || quote_ident(r.table_name), r.column_name);

        EXECUTE format('SELECT COALESCE(MAX(%I), 0) FROM proc.%I', r.column_name, r.table_name)
           INTO max_id;

        -- is_called lives on the sequence relation, not in pg_sequences, so the
        -- next value has to be read from the sequence itself.
        EXECUTE format('SELECT last_value, is_called FROM %s', seq_name)
           INTO last_val, called;
        next_val := last_val + CASE WHEN called THEN 1 ELSE 0 END;

        -- setval(..., max + 1, false) => the NEXT value handed out is max + 1.
        -- Only ever moves a sequence forward: a sequence already ahead of its
        -- data (normal, after deletes) is left alone.
        IF max_id >= next_val THEN
            PERFORM setval(seq_name, max_id + 1, false);
            fixed := fixed + 1;
            RAISE NOTICE 'resynced % -> next %', seq_name, max_id + 1;
        END IF;
    END LOOP;

    RAISE NOTICE 'identity sequences resynced: %', fixed;
END $$;
