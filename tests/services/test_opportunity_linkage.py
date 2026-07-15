from src.services.opportunity_linkage import link_opportunities_to_deals

class _Cur:
    def __init__(self): self.calls = []; self.rowcount = 3
    def execute(self, sql, params=()): self.calls.append(" ".join(sql.lower().split()))

class _Conn:
    def __init__(self): self._cur = _Cur(); self.autocommit = True
    def cursor(self): return self._cur
    def commit(self): pass
    def rollback(self): pass

def test_linkage_joins_quote_to_deal():
    c = _Conn()
    n = link_opportunities_to_deals(conn=c)
    s = c._cur.calls[0]
    assert "update proc.bp_opportunity" in s
    assert "bp_deal_documents" in s
    assert "doc_type = 'quote'" in s
    assert "o.deal_id  is null".replace("  ", " ") in s.replace("  ", " ")
    assert n == 3
