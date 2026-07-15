import importlib
import src.services.db as db
mod = importlib.import_module("src.api.routers.opportunities")

class _Cur:
    def __init__(self, rows): self._rows = rows
        # description drives column names -> dict rows
    description = [("opportunity_id",),("detector_type",),("category_id",),("supplier_name",),
                   ("item_description",),("financial_impact_gbp",),("stage",),
                   ("ml_priority_score",),("quote_id",)]
    def execute(self, sql, params=()): self.sql = sql; self.params = params
    def fetchall(self): return self._rows

class _Conn:
    def __init__(self, rows): self._c = _Cur(rows)
    def cursor(self): return self._c
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_by_deal_returns_rows(monkeypatch):
    rows = [("OPP-1","price_variance","cat-9","Acme","Widget",8100.0,"identified",0.9,"QA-1042")]
    conn = _Conn(rows)
    monkeypatch.setattr(db, "get_conn", lambda: conn)
    res = mod.get_opportunities_by_deal("ACME2026071501")
    assert res["deal_id"] == "ACME2026071501"
    assert res["opportunities"][0]["financial_impact_gbp"] == 8100.0
    assert res["opportunities"][0]["detector_type"] == "price_variance"
    sql = conn._c.sql.lower()
    assert "from proc.bp_opportunity" in sql
    assert "where deal_id = %s" in sql
    assert conn._c.params == ("ACME2026071501",)

def test_by_deal_empty_is_not_error(monkeypatch):
    conn = _Conn([])
    monkeypatch.setattr(db, "get_conn", lambda: conn)
    res = mod.get_opportunities_by_deal("NOPE")
    assert res["opportunities"] == []
    sql = conn._c.sql.lower()
    assert "from proc.bp_opportunity" in sql
    assert "where deal_id = %s" in sql
    assert conn._c.params == ("NOPE",)
