import importlib
from datetime import date

from src.services.structural_extractor.derivation import clear_registry
from src.services.structural_extractor.types import ExtractedValue


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


def _reload_dates():
    """Clear registry and reload the dates module so decorators re-run."""
    clear_registry()
    from src.services.structural_extractor.derivation_rules import dates
    importlib.reload(dates)
    return dates


def test_due_date_from_net_30():
    _reload_dates()
    from src.services.structural_extractor.derivation import resolve_all
    header = {
        "invoice_date": _ev(date(2020, 4, 1)),
        "payment_terms": _ev("Net 30"),
    }
    out = resolve_all(header, "Invoice")
    assert "due_date" in out
    assert out["due_date"].value == date(2020, 5, 1)
    assert out["due_date"].derivation_trace["rule_id"] == "due_date_from_terms"


def test_due_date_default_90_days():
    _reload_dates()
    from src.services.structural_extractor.derivation import resolve_all
    header = {"invoice_date": _ev(date(2020, 4, 1))}
    out = resolve_all(header, "Invoice")
    assert "due_date" in out
    assert out["due_date"].value == date(2020, 6, 30)  # +90 days
    assert out["due_date"].derivation_trace["rule_id"] == "due_date_default"


def test_due_date_from_within_14_days():
    _reload_dates()
    from src.services.structural_extractor.derivation import resolve_all
    header = {
        "invoice_date": _ev(date(2020, 4, 1)),
        "payment_terms": _ev("within 14 days"),
    }
    out = resolve_all(header, "Invoice")
    assert out["due_date"].value == date(2020, 4, 15)
