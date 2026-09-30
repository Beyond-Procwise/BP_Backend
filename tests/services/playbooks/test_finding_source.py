"""Two stores, two vocabularies, one shape -- and they never cross."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks.finding_source import (  # noqa: E402
    DETECTION_FINDING,
    OPPORTUNITY,
    MATCH_FIELDS,
    canonical,
    normalise,
    validate_trigger_match,
)


def test_detection_finding_row_normalises():
    finding = normalise(
        DETECTION_FINDING,
        {
            "finding_id": 4211,
            "rule_id": "cumulative_total",
            "category": "overbilling",
            "severity": "critical",
            "doc_type": "invoice",
            "blocks_promotion": True,
            "deal_id": "D-900",
            "notes": "ignored -- not a match field",
        },
    )
    assert finding.source == DETECTION_FINDING
    assert finding.finding_id == "4211"          # BIGINT arrives as int, stored as text
    assert finding.deal_id == "D-900"
    assert finding.attrs == {
        "rule_id": "cumulative_total",
        "category": "overbilling",
        "severity": "critical",
        "doc_type": "invoice",
        "blocks_promotion": True,
    }


def test_opportunity_row_normalises():
    finding = normalise(
        OPPORTUNITY,
        {
            "opportunity_id": "OPP-17",
            "detector_type": "Duplicate Invoice Recovery",
            "supplier_id": "SUP-3",
            "category_id": None,
            "deal_id": "D-900",
            "financial_impact_gbp": 1200,
        },
    )
    assert finding.finding_id == "OPP-17"
    assert finding.attrs == {
        "detector_type": "Duplicate Invoice Recovery",
        "supplier_id": "SUP-3",
        "category_id": None,
    }


def test_the_two_vocabularies_do_not_cross():
    """rule_id is not a field an opportunity playbook may match on, and
    detector_type is not one a detection-finding playbook may match on."""
    assert "rule_id" not in MATCH_FIELDS[OPPORTUNITY]
    assert "detector_type" not in MATCH_FIELDS[DETECTION_FINDING]
    with pytest.raises(ValueError) as exc:
        validate_trigger_match(OPPORTUNITY, {"rule_id": "quantity"})
    assert "rule_id" in str(exc.value)
    assert "detector_type" in str(exc.value)     # names what IS allowed


def test_unknown_source_is_refused():
    with pytest.raises(ValueError):
        normalise("invoices", {"finding_id": 1})
    with pytest.raises(ValueError):
        validate_trigger_match("invoices", {})


def test_unknown_match_key_is_refused_not_stored():
    """A key that matches no column is a playbook that silently never fires."""
    with pytest.raises(ValueError) as exc:
        validate_trigger_match(DETECTION_FINDING, {"severity": "critical", "sevrity": "high"})
    assert "sevrity" in str(exc.value)


def test_null_match_value_is_refused():
    """{"doc_type": null} would mean "match a finding whose doc_type is unset",
    which is never what an author means and always matches almost nothing."""
    with pytest.raises(ValueError) as exc:
        validate_trigger_match(DETECTION_FINDING, {"doc_type": None})
    assert "doc_type" in str(exc.value)


def test_empty_match_is_a_legitimate_catch_all():
    assert validate_trigger_match(DETECTION_FINDING, {}) == {}


def test_validate_returns_the_match_unchanged():
    match = {"rule_id": "quantity", "severity": "critical"}
    assert validate_trigger_match(DETECTION_FINDING, match) == match


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Critical", "critical"),
        ("critical", "critical"),
        (True, "true"),
        ("true", "true"),
        (False, "false"),
        (123, "123"),
        ("  padded  ", "padded"),
        (None, None),
    ],
)
def test_canonical_folds_both_sides_to_one_form(value, expected):
    """A boolean column compared against a JSON string, or a capitalised
    severity, must not silently fail to match."""
    assert canonical(value) == expected
