from src.services.structural_extractor.derivation import (
    DerivationRule, REGISTRY, rule, clear_registry, resolve_all
)
from src.services.structural_extractor.types import ExtractedValue


def _ev(value):
    return ExtractedValue(value=value, provenance="extracted", source="structural", attempt=1)


def test_registering_a_rule_adds_to_registry():
    clear_registry()

    @rule("test_rule", "target_x", ["input_a", "input_b"])
    def _fn(inputs):
        return inputs["input_a"] + inputs["input_b"]

    assert len(REGISTRY) == 1
    r = REGISTRY[0]
    assert r.rule_id == "test_rule"
    assert r.target_field == "target_x"
    assert r.inputs == ["input_a", "input_b"]
    assert r.compute({"input_a": 1, "input_b": 2}) == 3


def test_resolve_respects_dependencies():
    clear_registry()

    # subtotal = total - tax
    @rule("subtotal_rule", "invoice_amount", ["invoice_total_incl_tax", "tax_amount"])
    def _sub(inputs):
        return inputs["invoice_total_incl_tax"] - inputs["tax_amount"]

    # total = subtotal + tax (depends on subtotal, but also fires if total is missing)
    @rule("total_rule", "invoice_total_incl_tax", ["invoice_amount", "tax_amount"])
    def _tot(inputs):
        return inputs["invoice_amount"] + inputs["tax_amount"]

    # Start with total + tax; subtotal should be derived
    header = {"invoice_total_incl_tax": _ev(9999.60), "tax_amount": _ev(1666.60)}
    out = resolve_all(header, "Invoice")
    assert "invoice_amount" in out
    assert abs(out["invoice_amount"].value - 8333.00) < 0.01
    assert out["invoice_amount"].provenance == "derived"
    assert out["invoice_amount"].derivation_trace["rule_id"] == "subtotal_rule"


def test_resolve_does_not_overwrite_extracted():
    clear_registry()

    @rule("r", "invoice_amount", ["invoice_total_incl_tax", "tax_amount"])
    def _r(inputs):
        return inputs["invoice_total_incl_tax"] - inputs["tax_amount"]

    # Already extracted: must not overwrite
    header = {
        "invoice_amount": _ev(5000.0),  # already extracted (wrong but must be kept)
        "invoice_total_incl_tax": _ev(9999.60),
        "tax_amount": _ev(1666.60),
    }
    out = resolve_all(header, "Invoice")
    assert out["invoice_amount"].value == 5000.0  # unchanged
    assert out["invoice_amount"].provenance == "extracted"
