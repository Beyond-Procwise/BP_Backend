import src.services.agent_actions as aa


class _RecordingCursor:
    def __init__(self):
        self.executed = []
        self.many = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def executemany(self, sql, params):
        self.many.append((sql, list(params)))


class _RecordingConn:
    def __init__(self):
        self._cur = _RecordingCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_record_action_uses_injected_conn_and_does_not_commit():
    conn = _RecordingConn()
    aa.record_action(
        phase=aa.PHASE_EXTRACTION,
        action_type="regex_extract",
        doc_type="invoice",
        doc_pk="INV-1",
        details={"n": 3},
        conn=conn,
    )
    # The write is wrapped in a SAVEPOINT, so executed also holds SAVEPOINT/
    # RELEASE statements; find the one INSERT among them.
    inserts = [(sql, params) for sql, params in conn._cur.executed
               if "INSERT INTO proc.agent_actions" in sql]
    assert len(inserts) == 1
    sql, params = inserts[0]
    # details serialized to JSON text
    assert '"n": 3' in params[12]
    # a SAVEPOINT was taken and released around the insert
    stmts = [s for s, _ in conn._cur.executed]
    assert any(s.startswith("SAVEPOINT") for s in stmts)
    assert any(s.startswith("RELEASE SAVEPOINT") for s in stmts)
    # caller owns the transaction: writer must NOT commit an injected conn
    assert conn.committed is False


def test_record_action_swallows_db_errors(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(aa, "get_conn", boom)
    # Must not raise — best-effort logging cannot break the pipeline.
    aa.record_action(phase=aa.PHASE_EXTRACTION, action_type="persist")


def test_record_action_swallows_bad_phase_missing():
    # Missing required phase/action_type must be swallowed, not raised.
    aa.record_action(phase=None, action_type=None)  # type: ignore[arg-type]


def test_bulk_record_uses_executemany_on_injected_conn():
    conn = _RecordingConn()
    aa.bulk_record(
        [
            {"phase": aa.PHASE_VALIDATION, "action_type": "discrepancy", "field_name": "tax_amount"},
            {"phase": aa.PHASE_VALIDATION, "action_type": "discrepancy", "field_name": "total"},
        ],
        conn=conn,
    )
    assert len(conn._cur.many) == 1
    sql, rows = conn._cur.many[0]
    assert "INSERT INTO proc.agent_actions" in sql
    assert len(rows) == 2


def test_bulk_record_empty_is_noop():
    conn = _RecordingConn()
    aa.bulk_record([], conn=conn)
    assert conn._cur.many == []


class _FailingInsertCursor:
    """Cursor that records statements but raises on the agent_actions INSERT,
    simulating e.g. a missing table — to prove the SAVEPOINT contains the failure."""

    def __init__(self):
        self.executed = []

    def execute(self, sql, params=()):
        self.executed.append(sql)
        if "INSERT INTO proc.agent_actions" in sql:
            raise RuntimeError("relation does not exist")

    def executemany(self, sql, params):
        self.executed.append(sql)
        if "INSERT INTO proc.agent_actions" in sql:
            raise RuntimeError("relation does not exist")


class _FailingConn:
    def __init__(self):
        self._cur = _FailingInsertCursor()

    def cursor(self):
        return self._cur


def test_shared_conn_insert_failure_rolls_back_to_savepoint_and_is_swallowed():
    # A failed action insert on a caller-owned conn must (a) not propagate, and
    # (b) issue ROLLBACK TO SAVEPOINT so the caller's transaction stays usable.
    conn = _FailingConn()
    aa.record_action(
        phase=aa.PHASE_VALIDATION, action_type="discrepancy",
        field_name="tax_amount", conn=conn,
    )  # must not raise
    stmts = conn._cur.executed
    assert any(s.startswith("SAVEPOINT") for s in stmts)
    assert any(s.startswith("ROLLBACK TO SAVEPOINT") for s in stmts)
    # the savepoint was NOT released (the insert failed)
    assert not any(s.startswith("RELEASE SAVEPOINT") for s in stmts)


def test_shared_conn_bulk_insert_failure_rolls_back_to_savepoint():
    conn = _FailingConn()
    aa.bulk_record(
        [{"phase": aa.PHASE_VALIDATION, "action_type": "discrepancy"}],
        conn=conn,
    )  # must not raise
    stmts = conn._cur.executed
    assert any(s.startswith("SAVEPOINT") for s in stmts)
    assert any(s.startswith("ROLLBACK TO SAVEPOINT") for s in stmts)
