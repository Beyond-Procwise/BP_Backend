import importlib

from src.services.structural_extractor.derivation import clear_registry
from src.services.structural_extractor.types import ExtractedValue


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


def _reload_misc():
    clear_registry()
    from src.services.structural_extractor.derivation_rules import misc
    importlib.reload(misc)
    return misc


def test_country_from_uk_postcode():
    _reload_misc()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"_address_text": _ev("10 Redkiln Way Horsham RH13 5QH")}
    out = resolve_all(header, "Invoice")
    assert out["country"].value == "United Kingdom"


def test_country_from_us_zip():
    _reload_misc()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"_address_text": _ev("100 Main St, Springfield, IL 62701")}
    out = resolve_all(header, "Invoice")
    assert out["country"].value == "United States"


def test_invoice_status_default():
    _reload_misc()
    from src.services.structural_extractor.derivation import resolve_all
    out = resolve_all({"invoice_id": _ev("INV001")}, "Invoice")
    assert out["invoice_status"].value == "Issued"


def test_po_status_default():
    _reload_misc()
    from src.services.structural_extractor.derivation import resolve_all
    out = resolve_all({"po_id": _ev("PO001")}, "Purchase_Order")
    assert out["po_status"].value == "Open"
