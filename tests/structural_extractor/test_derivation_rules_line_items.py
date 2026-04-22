from src.services.structural_extractor.derivation_rules.line_items import (
    derive_line_item_fields,
)
from src.services.structural_extractor.types import ExtractedValue


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


def test_line_total_from_qty_price():
    item = {"quantity": _ev(10), "unit_price": _ev(79.99)}
    out = derive_line_item_fields(item)
    assert abs(out["line_total"].value - 799.90) < 0.01
    assert out["line_total"].provenance == "derived"


def test_unit_price_from_qty_and_total():
    item = {"quantity": _ev(10), "line_total": _ev(799.90)}
    out = derive_line_item_fields(item)
    assert abs(out["unit_price"].value - 79.99) < 0.01


def test_quantity_from_price_and_total():
    item = {"unit_price": _ev(79.99), "line_total": _ev(799.90)}
    out = derive_line_item_fields(item)
    assert out["quantity"].value == 10
