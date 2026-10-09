"""Only the newest read of a document may write _stg (promotion.promote).

On 2026-07-30 nineteen documents were read twice: an older pipeline at 18:25 missed every figure
and was held by a blocking finding; a fixed one at 19:41 read them right and was promoted. At
20:40 a dedup clean-up dismissed the older reads' blockers, which released them into promote(),
and each overwrote its newer, good _stg row with nothing (Meridia MCP-Q-7740 V3: 1,200,000 ->
NULL). A read whose document already has a newer promoted read is now superseded.
"""
from src.services.extraction import promotion


class _Cur:
    def __init__(self, newer):
        self.newer, self.sql, self.description = newer, [], None
        self._next = None

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if sql.startswith("SELECT * FROM"):
            self.description = [type("D", (), {"name": n})() for n in ("raw_id", "doc_pk_candidate", "promotion_status")]
            self._next = (38666, "MCP-Q-7740 (V3)", "discrepancy")
        elif "raw_id > %s" in sql:
            self._next = (self.newer,) if self.newer else None
        else:
            raise AssertionError("promote() went on to write after it should have stopped: " + sql[:60])

    def fetchone(self):
        return self._next


class _Conn:
    def __init__(self, cur):
        self.cur, self.autocommit, self.rolled_back = cur, True, False

    def cursor(self):
        return self.cur

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_an_older_read_never_overwrites_a_newer_promoted_one(monkeypatch):
    conn = _Conn(_Cur(newer=38691))
    monkeypatch.setattr(promotion, "get_conn", lambda: conn)
    out = promotion.promote(38666, "quote")
    assert out == {"ok": False, "doc_pk": "MCP-Q-7740 (V3)", "reason": "superseded_by_newer_read", "newer_raw_id": 38691}
    assert conn.rolled_back
    assert not any(s.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE")) for s in conn.cur.sql)


def test_the_check_asks_only_for_a_newer_read_that_was_promoted(monkeypatch):
    conn = _Cur(newer=None)
    monkeypatch.setattr(promotion, "get_conn", lambda: _Conn(conn))
    try:
        promotion.promote(38666, "quote")
    except AssertionError:
        pass                                   # no newer read: promote() carries on to write, as before
    q = next(s for s in conn.sql if "raw_id > %s" in s)
    assert "doc_pk_candidate = %s" in q and "promotion_status = 'promoted'" in q
