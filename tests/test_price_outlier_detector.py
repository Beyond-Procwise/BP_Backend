from services.price_outlier.detector import (
    Finding, build_peer_index, describe, peers_for,
)
from services.price_outlier.rule import OutlierSettings, assess


def _row(item, uom, ccy, price, doc="PO1"):
    return {"point_id": f"po:{price}", "item_description": item,
            "unit_of_measure": uom, "currency": ccy, "unit_price": price,
            "quantity": 1, "doc_id": doc}


def test_peer_index_groups_on_the_same_key_the_engine_matches_on():
    rows = [_row("  Widget  ", "Each", "gbp", 100.0, doc="PO1"),
            _row("widget", "each", "GBP", 102.0, doc="PO2"),
            _row("Widget", "box", "GBP", 500.0, doc="PO3")]
    index = build_peer_index(rows)
    assert index[("widget", "each", "GBP")] == [("PO1", 100.0), ("PO2", 102.0)]
    assert index[("widget", "box", "GBP")] == [("PO3", 500.0)]


def test_peers_exclude_the_line_s_own_document():
    rows = [_row("Widget", "each", "GBP", 100.0, doc="PO1"),
            _row("Widget", "each", "GBP", 900.0, doc="PO2")]
    index = build_peer_index(rows)
    assert peers_for(index, ("widget", "each", "GBP"), own_doc="PO1") == [900.0]


def test_peers_for_an_unknown_key_is_empty_not_an_error():
    assert peers_for({}, ("nothing", "each", "GBP"), own_doc=None) == []


def test_note_names_the_comparison_in_plain_english():
    verdict = assess(11.69, [1.17] * 18, OutlierSettings())
    note = describe(3, "A4 Ruled Notebook", 11.69, verdict)
    assert "line 3" in note
    assert "A4 Ruled Notebook" in note
    assert "11.69" in note
    assert "1.17" in note
    assert "18 comparable" in note
    assert "check the unit price and quantity" in note
