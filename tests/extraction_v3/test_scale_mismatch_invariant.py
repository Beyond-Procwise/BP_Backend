"""Tests for ScaleMismatch invariant (I-39 — TECHWORLD decimal-point misreads)."""
from src.services.extraction_v3.binding.scale_mismatch import ScaleMismatch


def test_scale_mismatch_triggers_on_10x_invoice():
    """The TECHWORLD I-39 case: line_sum=6750, header invoice_amount=675."""
    v = ScaleMismatch()
    result = v.check(
        header={"invoice_amount": 675.00},
        line_items=[
            {"line_amount": 5000.00},
            {"line_amount": 1750.00},
        ],
        doc_type="invoice",
    )
    assert result.severity in ("CRITICAL", "critical") or "CRITICAL" in str(result.severity).upper()
    assert "scale" in (result.message or "").lower() or "ratio" in (result.message or "").lower()


def test_scale_mismatch_does_not_trigger_on_consistent():
    """Consistent amounts should not raise CRITICAL."""
    v = ScaleMismatch()
    result = v.check(
        header={"invoice_amount": 6750.00},
        line_items=[
            {"line_amount": 5000.00},
            {"line_amount": 1750.00},
        ],
        doc_type="invoice",
    )
    # Consistent — should NOT be critical
    assert "CRITICAL" not in str(result.severity).upper()


def test_scale_mismatch_skips_when_data_missing():
    """Missing header or line_items should not CRITICAL — treat as non-applicable."""
    v = ScaleMismatch()
    result = v.check(header={}, line_items=[], doc_type="invoice")
    assert "CRITICAL" not in str(result.severity).upper()


def test_scale_mismatch_handles_quote():
    """10x mismatch on a quote should also be CRITICAL."""
    v = ScaleMismatch()
    result = v.check(
        header={"total_amount": 100},
        line_items=[{"line_amount": 1000}],  # 10×
        doc_type="quote",
    )
    assert "CRITICAL" in str(result.severity).upper() or "critical" in str(result.severity).lower()


def test_scale_mismatch_not_applicable_to_contract():
    """Contracts are not in the applicable set — should return a passing result."""
    v = ScaleMismatch()
    assert v.applicable("contract") is False
    result = v.check(
        header={"total_amount": 100},
        line_items=[{"line_amount": 1000}],
        doc_type="contract",
    )
    # check() returns ok() for non-applicable doc_types
    assert result.passed is True


def test_scale_mismatch_uses_fallback_header_field_invoice():
    """Falls back to invoice_total_incl_tax when invoice_amount is absent."""
    v = ScaleMismatch()
    result = v.check(
        header={"invoice_total_incl_tax": 100.00},
        line_items=[{"line_amount": 1000.00}],  # 10×
        doc_type="invoice",
    )
    assert "CRITICAL" in str(result.severity).upper()


def test_scale_mismatch_handles_purchase_order():
    """10x mismatch on a purchase_order should also be CRITICAL."""
    v = ScaleMismatch()
    result = v.check(
        header={"total_amount": 500.00},
        line_items=[{"line_amount": 50.00}],  # 10×
        doc_type="purchase_order",
    )
    assert "CRITICAL" in str(result.severity).upper()


def test_scale_mismatch_ratio_just_below_threshold():
    """Ratio of exactly 9 should NOT trigger CRITICAL (threshold is > 9)."""
    v = ScaleMismatch()
    result = v.check(
        header={"invoice_amount": 100.00},
        line_items=[{"line_amount": 900.00}],  # ratio = 9.0, not > 9
        doc_type="invoice",
    )
    assert "CRITICAL" not in str(result.severity).upper()


def test_scale_mismatch_uses_amount_key_fallback():
    """Line items keyed as 'amount' (not 'line_amount') are still summed."""
    v = ScaleMismatch()
    result = v.check(
        header={"invoice_amount": 50.00},
        line_items=[{"amount": 5000.00}],  # 100×
        doc_type="invoice",
    )
    assert "CRITICAL" in str(result.severity).upper()
