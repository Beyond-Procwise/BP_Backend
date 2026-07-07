"""T2: shared vendor_key derivation, and telemetry delegates to it."""
from src.services.extraction_feedback.vendor_key import vendor_key


def test_leading_supplier_token_before_marker():
    assert vendor_key("NEXASPARK INV4759276 for PO506789 (1).docx") == "NEXASPARK"
    assert vendor_key("documents/Quote/GOMEZ, GOOD ETC QUT104683 .pdf") == "GOMEZ, GOOD ETC"
    # No space/marker → first-word fallback keeps the whole basename (extension included),
    # matching the original telemetry behaviour exactly.
    assert vendor_key("BRAMWELL_QUOTE_208473.pdf") == "BRAMWELL_QUOTE_208473.pdf"


def test_first_word_fallback_and_none():
    assert vendor_key("Acme random file.pdf") == "Acme"
    assert vendor_key(None) is None
    assert vendor_key("") is None


def test_telemetry_delegates_identically():
    from src.services.extraction_telemetry.telemetry_service import _vendor_hint
    for p in [
        "NEXASPARK INV4759276 for PO506789 (1).docx",
        "GOMEZ, GOOD ETC QUT104683 .pdf",
        "Acme random file.pdf",
        None,
    ]:
        assert _vendor_hint(p) == vendor_key(p)
