"""Tests for the calibrated confidence score."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.confidence import calibrated_confidence  # noqa: E402


def _good_invoice():
    return {
        "invoice_id": "INV-1", "supplier_id": "SUP-1",
        "invoice_total_incl_tax": 100.0, "invoice_date": "2026-05-04",
    }


def test_clean_extraction_scores_high():
    res = calibrated_confidence(
        header=_good_invoice(),
        line_items=[{"item_description": "X", "quantity": 1, "line_amount": 100.0}],
        doc_type="Invoice",
    )
    assert res.score >= 0.95
    assert res.lines_invariant_pass is True
    assert res.required_fill == 1.0


def test_zero_lines_on_nonzero_total_drops_score():
    res = calibrated_confidence(
        header=_good_invoice(),
        line_items=[],
        doc_type="Invoice",
    )
    assert res.lines_invariant_pass is False
    assert res.score < 0.7  # multiplied by 0.65
    assert "zero_lines_on_nonzero_total" in res.notes


def test_zero_lines_with_zero_total_does_not_penalize():
    header = dict(_good_invoice(), invoice_total_incl_tax=0.0)
    res = calibrated_confidence(
        header=header, line_items=[], doc_type="Invoice",
    )
    assert res.lines_invariant_pass is True


def test_missing_required_field_drops_score():
    header = dict(_good_invoice())
    header.pop("invoice_id")
    res = calibrated_confidence(
        header=header,
        line_items=[{"item_description": "X", "quantity": 1, "line_amount": 100.0}],
        doc_type="Invoice",
    )
    assert res.required_fill == 0.75  # 3 of 4 present
    assert res.score < 0.85


def test_each_rescue_subtracts_005():
    res = calibrated_confidence(
        header=_good_invoice(),
        line_items=[{"item_description": "X", "quantity": 1, "line_amount": 100.0}],
        doc_type="Invoice",
        rescued_fields=["supplier_name", "supplier_id"],
    )
    assert res.rescue_penalty == 0.10


def test_each_sanitizer_rejection_subtracts_005():
    rejections = [
        SimpleNamespace(field="payment_terms", reason="garbage"),
        SimpleNamespace(field="buyer_name", reason="garbage"),
    ]
    res = calibrated_confidence(
        header=_good_invoice(),
        line_items=[{"item_description": "X", "quantity": 1, "line_amount": 100.0}],
        doc_type="Invoice",
        sanitizer_rejections=rejections,
    )
    assert res.sanitizer_penalty == 0.10


def test_template_overrides_add_capped_bonus():
    res = calibrated_confidence(
        header=_good_invoice(),
        line_items=[{"item_description": "X", "quantity": 1, "line_amount": 100.0}],
        doc_type="Invoice",
        template_overrides=["supplier_name", "supplier_id", "buyer_name"],
    )
    # Bonus is capped at 0.10
    assert res.template_bonus == 0.10


def test_score_is_clamped_to_unit_interval():
    """Many penalties + zero-line invariant fail must not produce
    negative scores."""
    rejections = [SimpleNamespace(field=f"f{i}", reason="x") for i in range(10)]
    res = calibrated_confidence(
        header={"invoice_id": "X"},  # only 1 of 4 required
        line_items=[],
        doc_type="Invoice",
        sanitizer_rejections=rejections,
        rescued_fields=["a", "b", "c"],
    )
    assert 0.0 <= res.score <= 1.0


def test_unknown_doc_type_still_returns_a_score():
    res = calibrated_confidence(
        header={"x": 1}, line_items=[], doc_type="Mystery",
    )
    assert 0.0 <= res.score <= 1.0
