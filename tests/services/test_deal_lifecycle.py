import pytest

import src.services.deal_lifecycle as deal_lifecycle
from src.services.deal_lifecycle import promote_deal, save_reference

class _Cur:
    def __init__(self): self.calls = []
    def execute(self, sql, params=()): self.calls.append((" ".join(sql.lower().split()), params))

class _Conn:
    def __init__(self): self._cur = _Cur(); self.autocommit = True; self.committed = False
    def cursor(self): return self._cur
    def commit(self): self.committed = True
    def rollback(self): pass

class _CurFails:
    """Cursor whose first execute() raises, to exercise the rollback branch."""
    def __init__(self): self.calls = []
    def execute(self, sql, params=()):
        raise RuntimeError("boom")

class _SelfManagedConn:
    """Fake connection returned by get_conn() when _with_txn manages it itself
    (conn=None). Supports `with get_conn() as own:` plus autocommit/commit/rollback."""
    def __init__(self, cur=None):
        self._cur = cur or _Cur()
        self.autocommit = True
        self.commit_calls = 0
        self.rollback_calls = 0
    def cursor(self): return self._cur
    def commit(self): self.commit_calls += 1
    def rollback(self): self.rollback_calls += 1
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

def test_promote_sets_tracked_and_advances_opps():
    c = _Conn()
    promote_deal("ACME2026071501", conn=c)
    sqls = [s for s, _ in c._cur.calls]
    assert any("insert into proc.bp_deal" in s and "is_tracked" in s for s in sqls)
    assert any("update proc.bp_opportunity set stage='negotiation'" in s
               and "where deal_id" in s and "stage='identified'" in s for s in sqls)
    # deal_id passed as a bound param, never interpolated
    assert any("ACME2026071501" in str(p) for _, p in c._cur.calls)

def test_save_reference_keeps_draft():
    c = _Conn()
    save_reference("ACME2026071501", conn=c)
    s = " ".join(x for x, _ in c._cur.calls)
    assert "is_saved_reference" in s
    assert "is_tracked=true" not in s  # must NOT promote

def test_promote_self_managed_conn_commits(monkeypatch):
    fake = _SelfManagedConn()
    monkeypatch.setattr(deal_lifecycle, "get_conn", lambda: fake)

    promote_deal("ACME01")

    assert fake.autocommit is False
    sqls = [s for s, _ in fake._cur.calls]
    assert any("insert into proc.bp_deal" in s and "is_tracked" in s for s in sqls)
    assert fake.commit_calls == 1
    assert fake.rollback_calls == 0

def test_save_reference_self_managed_conn_rolls_back_on_error(monkeypatch):
    fake = _SelfManagedConn(cur=_CurFails())
    monkeypatch.setattr(deal_lifecycle, "get_conn", lambda: fake)

    with pytest.raises(RuntimeError):
        save_reference("ACME01")

    assert fake.rollback_calls == 1
    assert fake.commit_calls == 0
