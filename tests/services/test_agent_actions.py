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
    assert len(conn._cur.executed) == 1
    sql, params = conn._cur.executed[0]
    assert "INSERT INTO proc.agent_actions" in sql
    # details serialized to JSON text
    assert '"n": 3' in params[12]
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
