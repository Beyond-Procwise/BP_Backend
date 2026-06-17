import src.services.requirement_service as rs


class _FakeCursor:
    def __init__(self, table_data=None):
        self._table_data = table_data or {}
        self.description = []
        self._rows = []
        self.executed = []
    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []
    def fetchall(self):
        return self._rows
    def fetchone(self):
        return self._rows[0] if self._rows else None
    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data=None):
        self.cur = _FakeCursor(table_data)
        self.committed = False
    def cursor(self):
        return self.cur
    def commit(self):
        self.committed = True
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_mint_requirement_id_format():
    rid = rs.mint_requirement_id("alice")
    assert rid.startswith("REQ-")
    parts = rid.split("-")
    assert len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 8


def test_evaluate_completeness_partial():
    req = {"title": "Laptops", "category": "IT"}
    score, missing = rs.evaluate_completeness(req, rs.DEFAULT_REQUIRED_FIELDS)
    assert missing == ["quantity", "needed_by_date", "delivery_location"]
    assert abs(score - 2 / 5) < 1e-6


def test_evaluate_completeness_full():
    req = {f: "x" for f in rs.DEFAULT_REQUIRED_FIELDS}
    score, missing = rs.evaluate_completeness(req, rs.DEFAULT_REQUIRED_FIELDS)
    assert missing == []
    assert score == 1.0


def test_persist_executes_upsert(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(rs, "get_conn", lambda: conn)
    rs.persist({"requirement_id": "REQ-1", "status": "complete", "title": "Laptops"})
    joined = " ".join(sql for sql, _ in conn.cur.executed).lower()
    assert "insert into proc.bp_requirement" in joined
    assert "on conflict (requirement_id) do update" in joined
    assert conn.committed is True


def test_get_requirement_returns_row(monkeypatch):
    data = {"proc.bp_requirement": (
        ["requirement_id", "status", "title"],
        [("REQ-1", "complete", "Laptops")],
    )}
    monkeypatch.setattr(rs, "get_conn", lambda: _FakeConn(data))
    row = rs.get_requirement("REQ-1")
    assert row == {"requirement_id": "REQ-1", "status": "complete", "title": "Laptops"}


def test_get_requirement_missing_returns_none(monkeypatch):
    monkeypatch.setattr(rs, "get_conn", lambda: _FakeConn({}))
    assert rs.get_requirement("REQ-x") is None


def test_seed_context_degrades_to_empty(monkeypatch):
    def _boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(rs, "get_conn", _boom)
    assert rs.seed_context("IT") == {}
