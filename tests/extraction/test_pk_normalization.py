"""Unit tests for canonical document-PK normalization.

Guards the fix for the duplicate-quote defect: a quote referenced bare in a
sibling PO document ("...QUT136586.pdf" → 136586) and the same quote extracted
from its own quote document ("136586" kept as "QUT136586") must resolve to ONE
canonical key, so they don't create two proc.bp_quote_stg rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.dispatch import normalize_doc_pk  # noqa: E402


def test_quote_id_strips_qut_prefix():
    assert normalize_doc_pk("quote", "QUT136586") == "136586"


def test_quote_id_strips_prefix_with_separator():
    assert normalize_doc_pk("quote", "QUT-2025-051") == "2025-051"


def test_quote_id_strips_prefix_keeps_hyphenated_remainder():
    assert normalize_doc_pk("quote", "QUT25-304-34") == "25-304-34"


def test_quote_id_already_bare_is_unchanged():
    assert normalize_doc_pk("quote", "136586") == "136586"


def test_quote_id_without_trailing_digits_is_left_alone():
    # No identifier remains after the prefix → don't mangle.
    assert normalize_doc_pk("quote", "QUOTE") == "QUOTE"


def test_invoice_id_prefix_is_preserved():
    # Only quote PKs are normalized; invoice_id keeps its INV form.
    assert normalize_doc_pk("invoice", "INV600259") == "INV600259"


def test_none_passes_through():
    assert normalize_doc_pk("quote", None) is None


def test_whitespace_is_trimmed():
    assert normalize_doc_pk("quote", "  QUT136586 ") == "136586"
