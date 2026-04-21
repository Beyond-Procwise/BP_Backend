from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.parsing.model import BBox


def test_extracted_value_extracted_provenance():
    ev = ExtractedValue(
        value=8333.0, provenance="extracted",
        anchor_text="Subtotal £8,333",
        anchor_ref=BBox(1, 100, 200, 300, 220),
        source="structural", confidence=1.0, attempt=1,
    )
    assert ev.provenance == "extracted"
    assert ev.derivation_trace is None


def test_extracted_value_derived_provenance():
    ev = ExtractedValue(
        value="2019-11-20", provenance="derived",
        derivation_trace={"rule_id": "due_date_default", "inputs": {"invoice_date": "2019-08-22"}},
        source="derivation_registry", confidence=1.0, attempt=1,
    )
    assert ev.provenance == "derived"
    assert ev.anchor_ref is None
    assert ev.derivation_trace["rule_id"] == "due_date_default"
