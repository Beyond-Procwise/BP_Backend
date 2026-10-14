"""Pure ledger rules. Spec §3, §7."""
from decimal import Decimal

import pytest

from src.services import value_ledger as vl


def test_amount_is_required_and_positive_for_money_outcomes():
    with pytest.raises(vl.LedgerError) as e:
        vl.validate_outcome("claimed", None, "GBP", None)
    assert e.value.code == "invalid_amount"
    with pytest.raises(vl.LedgerError):
        vl.validate_outcome("claimed", "0", "GBP", None)
    with pytest.raises(vl.LedgerError):
        vl.validate_outcome("claimed", "abc", "GBP", None)


def test_amount_is_rounded_to_pence_and_currency_upper_cased():
    assert vl.validate_outcome("avoided", "120.456", "gbp", None) == (Decimal("120.46"), "GBP")


def test_recovered_needs_an_evidence_reference():
    with pytest.raises(vl.LedgerError) as e:
        vl.validate_outcome("recovered", "50", "GBP", "   ")
    assert e.value.code == "evidence_required"
    assert vl.validate_outcome("recovered", "50", "GBP", "CN-123") == (Decimal("50.00"), "GBP")


def test_claim_dropped_carries_no_amount():
    assert vl.validate_outcome("claim_dropped", "99", "GBP", None) == (None, None)


def test_currency_must_be_three_letters():
    with pytest.raises(vl.LedgerError) as e:
        vl.validate_outcome("claimed", "10", "POUNDS", None)
    assert e.value.code == "invalid_currency"


def test_gbp_passes_through_with_no_rate():
    out = vl.convert_to_gbp(Decimal("10.00"), "GBP", None)
    assert out == {"amount_gbp": Decimal("10.00"), "fx_rate": None, "fx_as_of": None}


def test_foreign_amount_is_converted_and_the_rate_kept():
    rates = {"USD": 1.0, "GBP": 0.8, "_fetched_at": "2026-09-25T09:00:00+00:00"}
    out = vl.convert_to_gbp(Decimal("100.00"), "USD", rates)
    assert out["amount_gbp"] == Decimal("80.00")
    assert out["fx_rate"] == Decimal("0.8")
    assert out["fx_as_of"] == "2026-09-25T09:00:00+00:00"


def test_unconvertible_currency_keeps_native_amount_and_null_gbp():
    out = vl.convert_to_gbp(Decimal("100.00"), "NZD", {"USD": 1.0, "GBP": 0.8})
    assert out == {"amount_gbp": None, "fx_rate": None, "fx_as_of": None}


def test_current_state_is_the_latest_row_nobody_superseded():
    rows = [
        {"outcome_id": 1, "outcome_type": "claimed", "supersedes_id": None, "recorded_at": 1},
        {"outcome_id": 2, "outcome_type": "recovered", "supersedes_id": None, "recorded_at": 2},
        {"outcome_id": 3, "outcome_type": "recovered", "supersedes_id": 2, "recorded_at": 3},
    ]
    assert vl.current_state(rows)["outcome_id"] == 3
    assert vl.current_state([]) is None
