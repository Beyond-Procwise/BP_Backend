import importlib
import pytest

mod = importlib.import_module("src.services.deal_analysis_service")


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
    m = mod.compute_deal_metrics("DEAL-1", conn=object())
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
    m = mod.compute_deal_metrics("DEAL-1", conn=object())
    assert m["deal_value"] == 500
    assert m["price_change_pct"] is None
    assert m["volume_change_pct"] is None
    assert m["efficiency_score"] is None


def test_unknown_deal_returns_none(monkeypatch):
    monkeypatch.setattr(mod, "gather_deal_context", lambda deal_id, conn=None: None)
    assert mod.compute_deal_metrics("NOPE", conn=object()) is None
