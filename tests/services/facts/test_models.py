"""The load-bearing guarantee of this phase: a CommercialFact cannot exist
without provenance. Enforced in the type, not by convention — so every number
downstream provably originated in a document span."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.models import (  # noqa: E402
    CommercialFact,
    FactProvenance,
    ValidationState,
    ValueBasis,
)


def _prov(**over):
    base = dict(document_id="INV000109-1", doc_type="invoice",
                extraction_id="prov-1", field_path="line_items[0].unit_price",
                page=1, locator="bbox:10,20,30,40", verbatim_snippet="86.94")
    base.update(over)
    return FactProvenance(**base)


def _fact(**over):
    base = dict(fact_id="f-1", tenant_id="default", fact_type="line_unit_price",
                unit_price=Decimal("86.94"), currency="GBP", quantity=Decimal("2"),
                extended_value=Decimal("173.88"),
                measure_role="unit_rate", basis_uom="each",
                arithmetic_state="consistent",
                value_basis=ValueBasis.AS_SUPPLIED,
                validation_state=ValidationState.VALID,
                provenance=[_prov()])
    base.update(over)
    return CommercialFact(**base)


def test_a_fact_with_provenance_constructs():
    f = _fact()
    assert f.provenance and f.provenance[0].document_id == "INV000109-1"


def test_a_fact_with_no_provenance_cannot_be_constructed():
    with pytest.raises(ValidationError) as e:
        _fact(provenance=[])
    assert "provenance" in str(e.value).lower()


def test_a_fact_with_provenance_omitted_cannot_be_constructed():
    with pytest.raises(ValidationError):
        CommercialFact(fact_id="f-2", tenant_id="default", fact_type="x")


def test_provenance_requires_a_document_and_a_locator():
    with pytest.raises(ValidationError):
        _prov(document_id="")
    with pytest.raises(ValidationError):
        _prov(locator="")


def test_money_fields_are_decimal_not_float():
    """float unit prices reintroduce the rounding drift this whole programme
    exists to eliminate."""
    f = _fact()
    assert isinstance(f.unit_price, Decimal)
    assert isinstance(f.quantity, Decimal)


def test_extended_value_is_not_computed_by_the_model():
    """The model is a record, not a calculator. Anything derived is computed by
    a named deterministic engine that can be pointed at, not silently by a
    property nobody reviews."""
    f = _fact()
    assert not hasattr(f, "compute_extended_value")


def test_reason_codes_survive_onto_the_fact():
    f = _fact(uom="30 days from quote date", uom_normalised=None,
              reason_codes=["UOM_UNMAPPED"])
    assert f.uom == "30 days from quote date"
    assert f.uom_normalised is None
    assert "UOM_UNMAPPED" in f.reason_codes


def test_value_basis_is_a_closed_enum():
    assert {b.value for b in ValueBasis} == {
        "as_supplied", "baseline_corrected", "normalised"}
    with pytest.raises(ValidationError):
        _fact(value_basis="whatever")


def test_measure_role_is_a_closed_enum():
    """'Price' can mean a unit rate or a total; 'volume' can mean a count or a
    physical measure. The role says which, so a comparison can never put one
    supplier's unit rate against another's line total."""
    from src.services.facts.models import MeasureRole
    assert {r.value for r in MeasureRole} == {
        "unit_rate", "extended_line", "document_total",
        "quantity", "tax", "discount"}
    with pytest.raises(ValidationError):
        _fact(measure_role="price")


def test_a_unit_rate_must_declare_what_it_is_per():
    """£86.94 is not comparable to £86.94 until you know one is per user-month
    and the other per day. A unit_rate without a basis_uom is not a fact, it is
    a number."""
    with pytest.raises(ValidationError):
        _fact(measure_role="unit_rate", basis_uom=None)
    ok = _fact(measure_role="unit_rate", basis_uom="user_month")
    assert ok.basis_uom == "user_month"


def test_a_lump_sum_service_line_needs_no_unit_rate():
    """Services in this corpus legitimately have no unit price. The role says
    'this is a total', which is truthful — rather than a NULL unit_price that
    reads as missing data."""
    f = _fact(measure_role="extended_line", unit_price=None,
              quantity=None, basis_uom=None, extended_value=Decimal("58000.00"))
    assert f.measure_role.value == "extended_line"
    assert f.unit_price is None


def test_arithmetic_state_is_a_closed_enum_including_the_untestable_cases():
    """~19% of real lines have quantity=1, where a unit rate and a total are
    numerically identical and no arithmetic can separate them. The fact must
    record that its role was unverifiable rather than look identical to a
    checked one."""
    from src.services.facts.models import ArithmeticState
    assert {s.value for s in ArithmeticState} == {
        "consistent", "inconsistent",
        "untestable_quantity_one", "untestable_missing_input"}


def test_arithmetic_state_is_required_on_a_priced_fact():
    with pytest.raises(ValidationError):
        _fact(measure_role="unit_rate", basis_uom="each", arithmetic_state=None)


def test_concept_code_comes_from_the_extraction_schemas_not_an_invented_list():
    """B3 resolution: the extraction schemas ARE the field registry. Every
    concept_code must be a field name that actually exists in one of them —
    otherwise we have quietly created a second, shadow vocabulary, which is the
    failure the brief's 'do not shadow the dictionary' rule exists to prevent."""
    import yaml
    from pathlib import Path

    from src.services.facts.concept_codes import CONCEPT_CODES

    root = Path(__file__).resolve().parents[3] / "extraction_schemas"
    declared = set()
    for p in root.glob("*.yaml"):
        d = yaml.safe_load(p.read_text())
        declared |= {f["name"] for f in (d.get("fields") or [])}
        li = d.get("line_items") or {}
        declared |= {f["name"] for f in (li.get("fields") or [])}

    assert CONCEPT_CODES, "concept vocabulary is empty"
    orphans = CONCEPT_CODES - declared
    assert not orphans, f"concept codes with no schema field behind them: {sorted(orphans)}"


def test_concept_code_is_optional_on_a_fact():
    """A fact whose concept is not yet classified is still a valid fact. The
    code is an index into the registry, not a precondition for existing."""
    assert _fact().concept_code is None


def test_an_unknown_concept_code_is_rejected_rather_than_stored():
    """Silently accepting an unknown code is how a shadow vocabulary starts."""
    with pytest.raises(ValidationError):
        _fact(concept_code="not_a_real_field_name")
