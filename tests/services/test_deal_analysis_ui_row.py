import importlib
mod = importlib.import_module("src.services.deal_analysis_service")


def test_to_ui_row_formats_strings():
    row = {"deal_id": "DEAL-1", "supplier": "Acme", "category": "Electronics",
           "deal_value": 470000, "currency": "GBP", "volume": 105000,
           "unit_price": 4.48, "price_change_pct": 7.5, "volume_change_pct": -1.7,
           "efficiency_score": 18.06,
           "items": [{"name": "Widget A"}, {"name": "Bolt B"}]}
    ui = mod.to_ui_row(row)
    assert ui["id"] == "DEAL-1"
    assert ui["value"] == "£470K"
    assert ui["volume"] == "105,000"
    assert ui["unitPrice"] == "£4.48"
    assert ui["priceChange"] == "+7.5%"
    assert ui["volumeChange"] == "-1.7%"
    assert ui["efficiency"] == "18.06"
    assert ui["items"] == "Widget A, Bolt B"   # string, never a list


def test_to_ui_row_nulls_render_dash():
    row = {"deal_id": "DEAL-2", "supplier": None, "category": None,
           "deal_value": None, "currency": None, "volume": None,
           "unit_price": None, "price_change_pct": None, "volume_change_pct": None,
           "efficiency_score": None, "items": None}
    ui = mod.to_ui_row(row)
    assert ui["value"] == "–"
    assert ui["priceChange"] == "–"
    assert ui["items"] == "–"
