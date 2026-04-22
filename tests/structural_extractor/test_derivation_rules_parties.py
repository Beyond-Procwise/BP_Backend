import importlib

from src.services.structural_extractor.derivation import clear_registry
from src.services.structural_extractor.types import ExtractedValue


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


def _reload_parties():
    clear_registry()
    from src.services.structural_extractor.derivation_rules import parties
    importlib.reload(parties)
    return parties


def test_supplier_id_generated_from_name():
    parties = _reload_parties()
    # Set lookup callback to always-miss
    parties.set_supplier_lookup(lambda name: None)
    from src.services.structural_extractor.derivation import resolve_all
    header = {"supplier_name": _ev("City of Newport")}
    out = resolve_all(header, "Invoice")
    assert out["supplier_id"].value == "SUP-CITYOFNEWPORT"


def test_supplier_id_from_lookup_hit():
    parties = _reload_parties()
    parties.set_supplier_lookup(
        lambda name: "SUP-EXISTING-123" if "newport" in name.lower() else None
    )
    from src.services.structural_extractor.derivation import resolve_all
    header = {"supplier_name": _ev("City of Newport")}
    out = resolve_all(header, "Invoice")
    assert out["supplier_id"].value == "SUP-EXISTING-123"


def test_name_normalization():
    from src.services.structural_extractor.derivation_rules.parties import _normalize_name
    assert _normalize_name("  City of Newport  ") == "cityofnewport"
    assert _normalize_name("Duncan LLC") == "duncanllc"
    assert _normalize_name("Acme Corp.") == "acmecorp"
