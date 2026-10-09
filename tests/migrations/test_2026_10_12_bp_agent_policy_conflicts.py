"""Stage 4 conflict tables in both live databases. Needs PROCWISE_TEST_LIVE_DB=1.

Write tests seed their rows (tag below) inside a transaction that is always
rolled back, so nothing of theirs is ever left behind on a shared database.
"""
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")

# bp_decision columns as captured from the live DB (both databases) before this migration.
DECISION_COLUMNS = [
    "decision_id", "subject_type", "subject_id", "deal_id", "supplier_id", "decision",
    "resolution", "rationale", "policy_id", "policy_name", "facts", "evidence", "status",
    "actioned_by", "actioned_at", "override_reason", "workflow_id", "agent", "created_by",
    "created_at", "options", "respond_by", "on_timeout", "decision_scope", "levels",
    "current_level",
]
CONFLICT_COLUMNS = {
    "decision_id", "kind", "pair_key", "policy_keys", "policy_versions", "raised_by", "is_open",
    "outcome", "decided_by", "decided_at", "by_person", "created_at"}
RULE_COLUMNS = {
    "rule_id", "pair_key", "prevails", "yields", "rule_text", "decision_id", "decided_by",
    "decided_at", "superseded_at", "superseded_by"}
INDEXES = ("ix_bp_agent_policy_conflict_open_pair", "ix_bp_agent_policy_conflict_keys",
           "ix_bp_agent_policy_conflict_pair_decided", "ix_bp_agent_policy_conflict_rule_in_force")


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


class _Txn:
    """A transaction that is always rolled back; seeds decisions under a unique tag."""

    def __init__(self, dbname):
        self.conn = _connect(dbname)
        self.conn.autocommit = False
        self.cur = self.conn.cursor()
        self.tag = "t4mig-" + uuid.uuid4().hex[:10]

    def decision(self):
        self.cur.execute("INSERT INTO proc.bp_decision (subject_type, subject_id, decision,"
                         " created_by) VALUES ('policy_conflict', %s, 'open', %s)"
                         " RETURNING decision_id", (self.tag, self.tag))
        return self.cur.fetchone()[0]

    def conflict(self, pair, kind="policy", is_open=True, outcome=None, decided=False):
        d = self.decision()
        self.cur.execute(
            "INSERT INTO proc.bp_agent_policy_conflict (decision_id, kind, pair_key, policy_keys,"
            " policy_versions, raised_by, is_open, outcome, decided_at)"
            " VALUES (%s, %s, %s, ARRAY['A-0001','B-0001'], '{}', 'save', %s, %s,"
            " CASE WHEN %s THEN now() END)", (d, kind, pair, is_open, outcome, decided))
        return d

    def rule(self, pair, superseded=False):
        d = self.decision()
        self.cur.execute(
            "INSERT INTO proc.bp_agent_policy_conflict_rule (pair_key, prevails, yields, rule_text,"
            " decision_id, decided_by, decided_at, superseded_at)"
            " VALUES (%s, 'A-0001', 'B-0001', 'x', %s, %s, now(), CASE WHEN %s THEN now() END)",
            (pair, d, self.tag, superseded))

    def refused(self, exc, fn, *a, **k):
        import psycopg2
        self.cur.execute("SAVEPOINT sp")
        try:
            fn(*a, **k)
        except exc:
            self.cur.execute("ROLLBACK TO SAVEPOINT sp")
            return True
        except psycopg2.Error:
            raise
        self.cur.execute("RELEASE SAVEPOINT sp")
        return False

    def close(self):
        self.conn.rollback()
        self.conn.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_tables_columns_and_indexes_exist(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        assert set(_cols(cur, "bp_agent_policy_conflict")) == CONFLICT_COLUMNS
        assert set(_cols(cur, "bp_agent_policy_conflict_rule")) == RULE_COLUMNS
        for i in INDEXES:
            cur.execute("SELECT to_regclass(%s)", (f"proc.{i}",))
            assert cur.fetchone()[0] is not None, i


@pytest.mark.parametrize("dbname", DATABASES)
def test_open_pair_index_is_partial_unique(dbname):
    import psycopg2
    t = _Txn(dbname)
    try:
        t.conflict("P|Q")
        # a second OPEN policy row for the same pair is refused
        assert t.refused(psycopg2.errors.UniqueViolation, t.conflict, "P|Q")
        # a second live row is allowed
        t.conflict("P|Q", kind="live")
        t.conflict("P|Q", kind="live")
        # closed rows are allowed, many of them
        t.conflict("P|Q", is_open=False, outcome="a", decided=True)
        t.conflict("P|Q", is_open=False, outcome="a", decided=True)
    finally:
        t.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_one_rule_in_force_per_pair(dbname):
    import psycopg2
    t = _Txn(dbname)
    try:
        t.rule("R|S")
        assert t.refused(psycopg2.errors.UniqueViolation, t.rule, "R|S")
        t.rule("R|S", superseded=True)     # superseded rules are history; many allowed
        t.rule("R|S", superseded=True)
        t.rule("R|T")                      # another pair is independent
    finally:
        t.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_settled_needs_outcome_and_time(dbname):
    import psycopg2
    t = _Txn(dbname)
    try:
        assert t.refused(psycopg2.errors.CheckViolation, t.conflict, "U|V",
                         is_open=False, outcome=None, decided=True)
        assert t.refused(psycopg2.errors.CheckViolation, t.conflict, "U|V",
                         is_open=False, outcome="a", decided=False)
        t.conflict("U|V", is_open=False, outcome="a", decided=True)
    finally:
        t.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_notification_target_check(dbname):
    import psycopg2
    t = _Txn(dbname)
    sql = ("INSERT INTO proc.bp_policy_notification (firing_id, recipient, message, link)"
           " VALUES (NULL, %s, 'm', %s)")
    try:
        assert t.refused(psycopg2.errors.CheckViolation, t.cur.execute, sql,
                         (t.tag, "decision:1"))
        t.cur.execute(sql, (t.tag, "conflict:1"))
    finally:
        t.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_bp_decision_columns_unchanged(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        assert _cols(cur, "bp_decision") == DECISION_COLUMNS


@pytest.mark.parametrize("dbname", DATABASES)
def test_bp_policy_untouched(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='proc' AND table_name='bp_policy' ORDER BY 1")
        cols = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM proc.bp_policy")
        before = cur.fetchone()[0]
    assert cols == sorted(["created_by", "created_date", "last_modified_by", "last_modified_date",
                           "policy_desc", "policy_details", "policy_id", "policy_linked_agents",
                           "policy_name", "policy_status", "policy_type", "version"])
    # the write tests above ran in rolled-back transactions; the count must not move
    t = _Txn(dbname)
    try:
        t.conflict("W|X")
        t.cur.execute("SELECT count(*) FROM proc.bp_policy")
        assert t.cur.fetchone()[0] == before
    finally:
        t.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_no_seed_rows_left_behind(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_decision WHERE created_by LIKE 't4mig-%'")
        assert cur.fetchone()[0] == 0
