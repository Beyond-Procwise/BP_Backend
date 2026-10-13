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


class _WideCur(_Cur):
    """The real statement's shape: the nine listed columns, then the grouping ones."""
    def execute(self, sql, params=()):
        super().execute(sql, params)
        assert "supplier_id" in sql and "review_requested_at" in sql


def _wide(oid, lever, saving, stage="identified", asked=None, detail=None):
    return (oid, lever, "cat-9", "Acme", "Widget", saving, stage, 0.9, None,
            "SUP-1", "ITM-1", 12.0, 40, {}, ["PO-1"], None, asked, detail)


def test_by_deal_groups_levers_on_one_item_into_a_bucket(monkeypatch):
    import datetime as dt
    asked = dt.datetime(2026, 10, 13, tzinfo=dt.timezone.utc)
    changed = {"new_records": ["INV-2"], "impact_before": 300.0, "impact_after": 410.0}
    conn = _Conn([])
    conn._c = _WideCur([_wide("a", "Price Benchmark Variance", 410.0, "rejected", asked, changed),
                        _wide("b", "Volume Consolidation", 120.0)])
    monkeypatch.setattr(db, "get_conn", lambda: conn)

    res = mod.get_opportunities_by_deal("D1")

    (bucket,) = res["buckets"]
    assert bucket["leverCount"] == 2
    assert bucket["potentialSaving"] == 120.0          # the open lever, not 530
    assert {o["bucket_id"] for o in res["opportunities"]} == {bucket["bucketId"]}
    rejected = next(o for o in res["opportunities"] if o["opportunity_id"] == "a")
    assert rejected["review_requested"] is True and rejected["what_changed"] == changed
    assert "calculation_details" not in rejected and "supplier_id" not in rejected


def test_by_deal_rows_with_no_bucket_say_so(monkeypatch):
    conn = _Conn([("OPP-1", "Invoice Overbilling", None, "Acme", None, 70.0, "identified",
                   None, None, "SUP-1", None, None, None, {}, [], None, None, None)])
    monkeypatch.setattr(db, "get_conn", lambda: conn)

    res = mod.get_opportunities_by_deal("D1")

    assert res["buckets"] == []
    assert res["opportunities"][0]["bucket_id"] is None
    assert res["opportunities"][0]["review_requested"] is False
