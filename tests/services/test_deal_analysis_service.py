import importlib
import pytest

mod = importlib.import_module("src.services.deal_analysis_service")


class _FakeCur:
    """Minimal cursor so compute_deal_metrics gets a non-None cur."""
    def execute(self, sql, params=()):
        pass
    def fetchone(self):
        return None


class _FakeConn:
    def cursor(self):
        return _FakeCur()


def _ctx(invoices=None, pos=None, quotes=None, deal_name="DEAL-1"):
    return {
        "deal_id": "DEAL-1",
        "deal_name": deal_name,
        "documents": {
            "invoices": invoices or [],
            "purchase_orders": pos or [],
            "quotes": quotes or [],
        },
        "sources": {},
    }


def test_full_deal_metrics(monkeypatch):
    inv = {"supplier_name": "Acme", "total_amount": 1000, "currency": "GBP",
           "line_items": [
               {"item_description": "Widget A", "quantity": 100, "unit_price": 8.0},
               {"item_description": "Bolt B", "quantity": 100, "unit_price": 2.0}]}
    quote = {"supplier_name": "Acme", "total_amount": 900, "currency": "GBP",
             "line_items": [
                 {"item_description": "Widget A", "quantity": 120, "unit_price": 7.0},
                 {"item_description": "Bolt B", "quantity": 80, "unit_price": 1.5}]}
    monkeypatch.setattr(mod, "gather_deal_context",
                        lambda deal_id, conn=None: _ctx(invoices=[inv], quotes=[quote]))
    monkeypatch.setattr(mod, "_deal_category", lambda cur, deal_id: "Electronics")
    m = mod.compute_deal_metrics("DEAL-1", conn=_FakeConn())
    assert m["supplier"] == "Acme"
    assert m["category"] == "Electronics"
    assert m["deal_value"] == 1000          # invoice total
    assert m["currency"] == "GBP"
    assert m["volume"] == 200               # 100 + 100 invoiced qty
    assert m["unit_price"] == pytest.approx(5.0)   # 1000 / 200
    # invoice weighted unit = 1000/200 = 5.0 ; quote weighted = (120*7+80*1.5)/200 = 4.8
    assert m["price_change_pct"] == pytest.approx(4.17, abs=0.01)   # (5.0-4.8)/4.8*100
    assert m["volume_change_pct"] == pytest.approx(0.0)             # 200 vs 200
    assert m["item_count"] == 2
    assert {i["name"] for i in m["items"]} == {"Widget A", "Bolt B"}


def test_missing_quote_leaves_changes_null(monkeypatch):
    inv = {"supplier_name": "Acme", "total_amount": 500, "currency": "USD",
           "line_items": [{"item_description": "X", "quantity": 50, "unit_price": 10.0}]}
    monkeypatch.setattr(mod, "gather_deal_context",
                        lambda deal_id, conn=None: _ctx(invoices=[inv]))
    monkeypatch.setattr(mod, "_deal_category", lambda cur, deal_id: None)
    m = mod.compute_deal_metrics("DEAL-1", conn=_FakeConn())
    assert m["deal_value"] == 500
    assert m["price_change_pct"] is None
    assert m["volume_change_pct"] is None
    assert m["efficiency_score"] is None


def test_unknown_deal_returns_none(monkeypatch):
    monkeypatch.setattr(mod, "gather_deal_context", lambda deal_id, conn=None: None)
    assert mod.compute_deal_metrics("NOPE", conn=object()) is None


def test_a_quotes_only_deal_stands_on_its_lowest_current_bid():
    # Three suppliers, two rounds each. The deal's value used to be all six totals added
    # (and its items every version's lines); its supplier whichever quote came first.
    from src.services.deal_analysis_service import _compute
    def q(qid, sup, amt, item):
        return {"quote_id": qid, "supplier_id": sup, "supplier_name": sup, "total_amount": amt,
                "currency": "GBP", "line_items": [{"item_description": item, "quantity": 1, "unit_price": amt}]}
    ctx = {"deal_id": "D1", "deal_name": "x", "documents": {"invoices": [], "purchase_orders": [], "quotes": [
        q("A-1", "SA", 100.0, "old A"), q("A-1 (V2)", "SA", 90.0, "new A"),
        q("B-1", "SB", 95.0, "old B"), q("B-1 (V2)", "SB", 92.0, "new B"),
        q("C-1", "SC", 80.0, "old C"), q("C-1 (V2)", "SC", 99.0, "new C"),
    ]}}
    out = _compute(ctx, None)
    assert out["deal_value"] == 90.0                 # A's current bid; C's old 80 is withdrawn
    assert out["supplier"] is None                   # three bidders, no award yet
    assert [i["name"] for i in out["items"]] == ["new A"]
