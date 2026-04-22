import importlib

from src.services.structural_extractor.derivation import clear_registry
from src.services.structural_extractor.types import ExtractedValue


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


def test_xrate_gbp_to_usd(monkeypatch):
    import src.services.structural_extractor.derivation_rules.fx as fx
    clear_registry()
    importlib.reload(fx)
    # Clear cache to avoid pollution between tests
    fx._CACHE.clear()
    monkeypatch.setattr(fx, "_fetch_rate_live", lambda ccy: 1.25)
    from src.services.structural_extractor.derivation import resolve_all
    header = {"currency": _ev("GBP")}
    out = resolve_all(header, "Invoice")
    assert "exchange_rate_to_usd" in out
    assert abs(out["exchange_rate_to_usd"].value - 1.25) < 0.001


def test_converted_amount_usd(monkeypatch):
    import src.services.structural_extractor.derivation_rules.fx as fx
    clear_registry()
    importlib.reload(fx)
    fx._CACHE.clear()
    monkeypatch.setattr(fx, "_fetch_rate_live", lambda ccy: 1.25)
    from src.services.structural_extractor.derivation import resolve_all
    header = {
        "currency": _ev("GBP"),
        "invoice_total_incl_tax": _ev(9999.60),
    }
    out = resolve_all(header, "Invoice")
    assert "converted_amount_usd" in out
    assert abs(out["converted_amount_usd"].value - 12499.50) < 0.01
