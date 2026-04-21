import importlib

from src.services.structural_extractor.derivation import clear_registry
from src.services.structural_extractor.types import ExtractedValue


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


def _reload_amounts():
    clear_registry()
    from src.services.structural_extractor.derivation_rules import amounts
    importlib.reload(amounts)
    return amounts


def test_subtotal_derived_from_total_minus_tax():
    _reload_amounts()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"invoice_total_incl_tax": _ev(9999.60), "tax_amount": _ev(1666.60)}
    out = resolve_all(header, "Invoice")
    assert abs(out["invoice_amount"].value - 8333.00) < 0.01


def test_tax_amount_from_subtotal_and_pct():
    _reload_amounts()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"invoice_amount": _ev(8333.00), "tax_percent": _ev(20.0)}
    out = resolve_all(header, "Invoice")
    assert abs(out["tax_amount"].value - 1666.60) < 0.01


def test_total_from_subtotal_plus_tax():
    _reload_amounts()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"invoice_amount": _ev(8333.00), "tax_amount": _ev(1666.60)}
    out = resolve_all(header, "Invoice")
    assert abs(out["invoice_total_incl_tax"].value - 9999.60) < 0.01


def test_tax_pct_from_amounts():
    _reload_amounts()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"invoice_amount": _ev(8333.00), "tax_amount": _ev(1666.60)}
    out = resolve_all(header, "Invoice")
    assert abs(out["tax_percent"].value - 20.0) < 0.1
