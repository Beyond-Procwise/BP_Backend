"""The variance derivation must respect BOTH conventions in computed_value.

bp_extraction_discrepancy stores computed_value two different ways:

  sum_mismatch / tax_percent_mismatch  -> an ABSOLUTE value
       expected 440.00, computed 460.0     => variance 20.00

  amount_over_po / line_amount_over_po -> ALREADY THE DELTA, explicitly signed
       expected 28610.00, computed +950.00 => variance 950.00

The first cut of the engine treated every row as absolute, so it computed
|950 - 28610| = 27,660 for a finding whose real over-billing is 950 — wrong by a
factor of 29, and wrong in the direction that makes a trivial exception look like a
catastrophe. That inflated figure then drove the escalate/approve call and was
written into the audit trail as fact.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engines.decision_engine import DecisionEngine


_POLICY = {
    "policyName": "ApprovalThresholdPolicy",
    "details": {"rules": {"default_threshold_gbp": 10000, "currency": "GBP"}},
    "raw_row": {"policy_id": 10, "policy_name": "ApprovalThresholdPolicy"},
}


def _engine(row):
    nick = SimpleNamespace(
        policy_engine=SimpleNamespace(get_policy=lambda slug: _POLICY),
        get_db_connection=MagicMock(side_effect=RuntimeError("db off in test")),
    )
    eng = DecisionEngine(nick)
    eng._fetch_finding = lambda _id: row  # type: ignore[assignment]
    return eng


def _row(**over):
    base = {
        "discrepancy_id": 1,
        "doc_type": "invoice",
        "source_file": "documents/invoice/x.pdf",
        "field_name": "invoice_amount",
        "raw_value": None,
        "expected_value": None,
        "computed_value": None,
        "issue_type": "sum_mismatch",
        "severity": "warning",
        "status": "open",
        "blocks_promotion": False,
        "evidence_page": None,
        "evidence_text": None,
    }
    base.update(over)
    return base


def test_absolute_convention_subtracts():
    """sum_mismatch: computed is an absolute value."""
    d = _engine(_row(expected_value="440.00", computed_value="460.0")).decide_finding("1")
    assert d.facts["variance"] == "20.00"


def test_signed_delta_convention_is_taken_as_the_delta():
    """amount_over_po: computed is ALREADY the delta. This is the regression."""
    d = _engine(
        _row(
            issue_type="amount_over_po",
            expected_value="28610.00",
            computed_value="+950.00",
        )
    ).decide_finding("1")
    assert d.facts["variance"] == "950.00", (
        "a signed computed_value IS the delta; subtracting it from expected gives "
        "27,660 for a 950 over-billing"
    )


def test_negative_delta_is_taken_as_its_magnitude():
    d = _engine(
        _row(
            issue_type="line_amount_over_po",
            expected_value="1000.00",
            computed_value="-105.00",
        )
    ).decide_finding("1")
    assert d.facts["variance"] == "105.00"


def test_the_inflated_variance_no_longer_flips_the_decision():
    """A 950 over-billing on a warning must APPROVE, not escalate on a phantom 27,660."""
    d = _engine(
        _row(
            issue_type="amount_over_po",
            severity="warning",
            expected_value="28610.00",
            computed_value="+950.00",
        )
    ).decide_finding("1")
    # 950 <= 10,000 threshold -> within delegated authority.
    assert d.decision == "approve"
    assert d.resolution == "resolved"


def test_critical_still_escalates_whatever_the_amount():
    d = _engine(
        _row(
            issue_type="amount_over_po",
            severity="critical",
            expected_value="28610.00",
            computed_value="+950.00",
        )
    ).decide_finding("1")
    assert d.decision == "escalate"


def test_non_numeric_values_yield_no_variance():
    d = _engine(
        _row(expected_value="ACME LTD", computed_value="ACME LIMITED")
    ).decide_finding("1")
    assert "variance" not in d.facts
    assert d.decision == "investigate"


def test_the_derivation_is_stated_in_the_evidence():
    """Whoever checks this must be able to see WHICH convention was applied."""
    d = _engine(
        _row(
            issue_type="amount_over_po",
            expected_value="28610.00",
            computed_value="+950.00",
        )
    ).decide_finding("1")
    src = next(e.source for e in d.evidence if e.fact == "variance")
    assert "already the delta" in src
