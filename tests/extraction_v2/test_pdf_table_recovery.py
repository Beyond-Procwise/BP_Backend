"""Tests for the pdfplumber-based line-items recovery.

These tests focus on the logic functions (column splitter, math check)
because the full pdfplumber integration test requires a sample PDF.
The integration is exercised by re-extracting WADE quote QUT30746 in
production after the recovery is wired in.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.pdf_table_recovery import (  # noqa: E402
    _split_row_into_columns, _row_looks_like_line_item,
    recover_lines_from_pdf_table,
)


def test_split_row_picks_item_id_qty_unitprice_total():
    """Standard 5-column row: ID, Description, Qty, UnitPrice, Total."""
    tokens = [
        "STA5183", "Staedtler", "Ballpoint", "Pen,", "Black", "Ink",
        "10", "£14.14", "£141.40",
    ]
    row = _split_row_into_columns(tokens)
    assert row["item_id"] == "STA5183"
    assert "Staedtler" in row["description"]
    assert row["quantity"] == 10
    assert row["unit_price"] == 14.14
    assert row["line_total"] == 141.40


def test_split_row_only_total_when_one_amount():
    """4-column row without unit-price column: only line_total emitted."""
    tokens = ["FAB1748", "Faber-Castell", "A4", "Ruled", "100", "£134.80"]
    row = _split_row_into_columns(tokens)
    assert row["item_id"] == "FAB1748"
    assert row["quantity"] == 100
    assert row["unit_price"] is None
    assert row["line_total"] == 134.80


def test_split_row_no_amounts_returns_none():
    """Non-line-item rows have no trailing amounts."""
    tokens = ["Item", "ID", "Description", "Qty"]
    assert _split_row_into_columns(tokens) is None


def test_split_row_returns_none_on_empty():
    assert _split_row_into_columns([]) is None


def test_no_item_id_when_first_token_is_just_a_word():
    """First token must be alphanumeric WITH digits to be an item ID;
    otherwise it's part of description."""
    tokens = ["Service", "Package", "1", "£50.00"]
    row = _split_row_into_columns(tokens)
    assert row["item_id"] is None
    assert row["description"] == "Service Package"
    assert row["quantity"] == 1
    assert row["line_total"] == 50.0


def test_row_filter_passes_when_math_consistent():
    row = {"item_id": "X", "description": "thing",
           "quantity": 10, "unit_price": 14.14, "line_total": 141.40}
    assert _row_looks_like_line_item(row, target=864.15, full_row_text="X thing 10 £14.14 £141.40") is True
    # qty/unit preserved
    assert row["quantity"] == 10
    assert row["unit_price"] == 14.14


def test_row_filter_drops_qty_unit_when_math_mismatches():
    """When qty × unit_price != line_total, drop qty/unit (column-detection
    likely picked up wrong tokens) but keep line_total — the user said
    'do not manipulate false data', so we'd rather have NULL than wrong."""
    row = {"item_id": "X", "description": "thing",
           "quantity": 100, "unit_price": 14.14, "line_total": 141.40}  # 100*14.14=1414
    assert _row_looks_like_line_item(row, target=864.15) is True
    assert row["quantity"] is None
    assert row["unit_price"] is None
    assert row["line_total"] == 141.40  # preserved


def test_row_filter_rejects_when_no_description():
    row = {"item_id": None, "description": None,
           "quantity": 5, "unit_price": 10.0, "line_total": 50.0}
    assert _row_looks_like_line_item(row, target=100.0, full_row_text="") is False


def test_row_filter_rejects_amount_too_large():
    """An amount > 2× target can't be a line item (it's the grand total)."""
    row = {"item_id": "X", "description": "thing",
           "quantity": None, "unit_price": None, "line_total": 1000.0}
    assert _row_looks_like_line_item(row, target=100.0, full_row_text="X thing £1000") is False


def test_row_filter_rejects_when_label_in_full_row():
    """Row 'Payment must be made within SUBTOTAL: £6,750' has desc
    'Payment must be made within' (no label) but the FULL row contains
    'SUBTOTAL:' — must reject so the £6,750 isn't double-counted."""
    row = {"item_id": None, "description": "Payment must be made within",
           "quantity": 30, "unit_price": None, "line_total": 6750.0}
    full = "Payment must be made within SUBTOTAL: £6750"
    assert _row_looks_like_line_item(row, target=6750.0, full_row_text=full) is False


def test_recover_skips_when_no_pdf_bytes():
    r = recover_lines_from_pdf_table(b"", target_total=100.0)
    assert r.items_recovered == 0
    assert r.skipped_reason == "no_pdf_bytes"


def test_recover_skips_when_no_table_rows():
    """Garbage PDF bytes — pdfplumber may either error or find nothing.
    Either way the report should indicate no recovery."""
    # Minimal valid-looking but content-less PDF stub
    pdf_stub = (
        b"%PDF-1.4\n"
        b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        b"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 100 100]>> endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"trailer <</Size 4 /Root 1 0 R>>\nstartxref\n0\n%%EOF\n"
    )
    r = recover_lines_from_pdf_table(pdf_stub, target_total=100.0)
    # Either no rows, or sum mismatch — both acceptable, both result in 0 items
    assert r.items_recovered == 0
