"""Stage 3 enforcement tables in both live databases. Needs PROCWISE_TEST_LIVE_DB=1."""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")

# bp_decision columns as captured from the live DB before this migration.
DECISION_EXISTING = [
    "decision_id", "subject_type", "subject_id", "deal_id", "supplier_id", "decision",
    "resolution", "rationale", "policy_id", "policy_name", "facts", "evidence", "status",
    "actioned_by", "actioned_at", "override_reason", "workflow_id", "agent", "created_by",
    "created_at",
]
DECISION_NEW = ["options", "respond_by", "on_timeout", "decision_scope", "levels", "current_level"]


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=dbname, connect_timeout=10,
    )


def _cols(cur, table):
    cur.execute("SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='proc' AND table_name=%s ORDER BY ordinal_position", (table,))
    return [r[0] for r in cur.fetchall()]


@pytest.mark.parametrize("dbname", DATABASES)
def test_tables_and_indexes_exist(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        for t in ("bp_policy_decider_map", "bp_policy_firing", "bp_policy_notification"):
            cur.execute("SELECT to_regclass(%s)", (f"proc.{t}",))
            assert cur.fetchone()[0] is not None, t
        for i in ("ix_bp_policy_firing_policy", "ix_bp_policy_firing_decision",
                  "ix_bp_policy_notification_recipient", "ix_bp_decision_open_respond_by"):
            cur.execute("SELECT to_regclass(%s)", (f"proc.{i}",))
            assert cur.fetchone()[0] is not None, i


@pytest.mark.parametrize("dbname", DATABASES)
def test_firing_columns(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        assert set(_cols(cur, "bp_policy_firing")) >= {
            "firing_id", "policy_key", "policy_version", "checkpoint", "action_name", "agent",
            "workflow_id", "requested_by", "outcome", "result", "matched_values",
            "missing_inputs", "decision_id", "decided_level", "decided_by", "decided_at",
            "reason", "duration_ms", "reversal_of", "created_at"}


@pytest.mark.parametrize("dbname", DATABASES)
def test_bp_decision_existing_columns_unchanged_and_new_are_nullable(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cols = _cols(cur, "bp_decision")
        assert cols[:len(DECISION_EXISTING)] == DECISION_EXISTING
        assert set(DECISION_NEW) <= set(cols)
        cur.execute("SELECT column_name, is_nullable FROM information_schema.columns "
                    "WHERE table_schema='proc' AND table_name='bp_decision' "
                    "AND column_name = ANY(%s)", (DECISION_NEW,))
        assert all(n == "YES" for _, n in cur.fetchall())


@pytest.mark.parametrize("dbname", DATABASES)
def test_decider_map_needs_someone(dbname):
    import psycopg2
    conn = _connect(dbname)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, last_modified_by)"
                        " VALUES ('__t__', 'test')")
        conn.rollback()
        cur = conn.cursor()
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, last_modified_by)"
                    " VALUES ('__t__', ARRAY['g'], 'test')")
    finally:
        conn.rollback()
        conn.close()


def _insert_firing(cur, result):
    cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint,"
                " action_name, outcome, result) VALUES ('P-0001', 1, 'cp', 'tool', 'approve', %s)"
                " RETURNING firing_id", (result,))
    return cur.fetchone()[0]


def _refused(cur, sql, args=()):
    import psycopg2
    cur.execute("SAVEPOINT sp")
    try:
        cur.execute(sql, args)
    except psycopg2.errors.RaiseException:
        cur.execute("ROLLBACK TO SAVEPOINT sp")
        return True
    cur.execute("RELEASE SAVEPOINT sp")
    return False


@pytest.mark.parametrize("dbname", DATABASES)
def test_firing_trigger_is_append_only(dbname):
    conn = _connect(dbname)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        paused = _insert_firing(cur, "paused_for_approval")
        done = _insert_firing(cur, "allowed")
        # DELETE refused, on any row
        assert _refused(cur, "DELETE FROM proc.bp_policy_firing WHERE firing_id=%s", (paused,))
        assert _refused(cur, "DELETE FROM proc.bp_policy_firing WHERE firing_id=%s", (done,))
        # a non-paused row cannot change at all
        assert _refused(cur, "UPDATE proc.bp_policy_firing SET reason='x' WHERE firing_id=%s", (done,))
        # a paused row: non-decision columns refused
        for col, val in (("policy_key", "'Z'"), ("action_name", "'Z'"), ("agent", "'Z'"),
                         ("matched_values", "'{\"a\":1}'"), ("outcome", "'block'"),
                         ("duration_ms", "5")):
            assert _refused(cur, f"UPDATE proc.bp_policy_firing SET {col}={val} WHERE firing_id=%s",
                            (paused,)), col
        # a paused row: decision columns allowed
        cur.execute("UPDATE proc.bp_policy_firing SET decision_id=7, decided_level=1,"
                    " decided_by='a@b.c', decided_at=now(), reason='ok', result='approved'"
                    " WHERE firing_id=%s", (paused,))
        # once resolved it is frozen
        assert _refused(cur, "UPDATE proc.bp_policy_firing SET reason='again' WHERE firing_id=%s",
                        (paused,))
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_bp_policy_untouched(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='proc' AND table_name='bp_policy' ORDER BY 1")
        cols = [r[0] for r in cur.fetchall()]
    assert cols == sorted(["created_by", "created_date", "last_modified_by", "last_modified_date",
                           "policy_desc", "policy_details", "policy_id", "policy_linked_agents",
                           "policy_name", "policy_status", "policy_type", "version"])
