from src.services.deal_lifecycle import promote_deal, save_reference

class _Cur:
    def __init__(self): self.calls = []
    def execute(self, sql, params=()): self.calls.append((" ".join(sql.lower().split()), params))

class _Conn:
    def __init__(self): self._cur = _Cur(); self.autocommit = True; self.committed = False
    def cursor(self): return self._cur
    def commit(self): self.committed = True
    def rollback(self): pass

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
