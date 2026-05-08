"""Tests for primary-key normalization."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.pk_normalizer import normalize_pk  # noqa: E402


def test_strips_inv_prefix_from_invoice_id():
    assert normalize_pk("INV148769", "Invoice") == "148769"
    assert normalize_pk("inv148769", "Invoice") == "148769"


def test_strips_invoice_prefix():
    assert normalize_pk("INVOICE-148769", "Invoice") == "148769"


def test_strips_po_prefix():
    assert normalize_pk("PO526689", "Purchase_Order") == "526689"


def test_strips_quote_prefix():
    assert normalize_pk("Q599390", "Quote") == "599390"
    assert normalize_pk("QUT599390", "Quote") == "599390"


def test_keeps_already_canonical():
    assert normalize_pk("148769", "Invoice") == "148769"
    assert normalize_pk("526689", "Purchase_Order") == "526689"


def test_keeps_vendor_token_with_separators():
    """`DHA-2025-143` and `QUT-25-032` are vendor-formatted IDs that
    must not be mangled — they carry meaning beyond just digits."""
    assert normalize_pk("DHA-2025-143", "Invoice") == "DHA-2025-143"
    assert normalize_pk("QUT-25-032", "Quote") == "QUT-25-032"


def test_keeps_vendor_token_with_no_recognised_prefix():
    assert normalize_pk("ABC-12345", "Invoice") == "ABC-12345"


def test_does_not_strip_short_digit_run():
    # Less than 4 digits — keep prefix to avoid collisions
    assert normalize_pk("INV12", "Invoice") == "INV12"


def test_handles_empty_and_none():
    assert normalize_pk("", "Invoice") == ""
    assert normalize_pk(None, "Invoice") == ""  # type: ignore[arg-type]


def test_trims_whitespace():
    assert normalize_pk("  INV148769  ", "Invoice") == "148769"


def test_does_not_strip_inv_when_doc_type_is_quote():
    """Cross-type prefix protection: don't strip 'INV' from a quote."""
    assert normalize_pk("INV148769", "Quote") == "INV148769"
