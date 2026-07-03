"""Regression tests for the version-aware contract-total recovery.

Covers the freight/rate-card quote class (e.g. Condor Logistics V1) where
docling detaches the grand-total label from its value in full_text and the
LLM mis-picks a section sub-total. The recovery reads THIS version's figure
from the structured "Total Value"/"Contract Value" summary table.
"""
from dataclasses import dataclass

from src.services.extraction.context_layer import (
    _doc_version,
    _recover_contract_total,
)


@dataclass
class _Cell:
    text: str
    col_index: int
    row_index: int = 0


class _Table:
    def __init__(self, rows, header_row_index=0):
        self.rows = rows
        self.header_row_index = header_row_index


def _version_history_table():
    # Row 0 is a spanned title ("VERSION HISTORY") — header_row_index points
    # here, but the real column header ("Total Value") is on row 1.
    return _Table(
        rows=[
            [_Cell("VERSION HISTORY", 0), _Cell("VERSION HISTORY", 1), _Cell("VERSION HISTORY", 2)],
            [_Cell("Version / Date", 0), _Cell("Total Value", 1), _Cell("Summary of Changes", 2)],
            [_Cell("V1 | 8 March 2024", 0), _Cell("£238,432", 1), _Cell("Initial rate card", 2)],
            [_Cell("V2 | 22 March 2024", 0), _Cell("£230,930", 1), _Cell("2% reduction", 2)],
            [_Cell("V3 (BAFO) | 5 April 2024", 0), _Cell("£219,000", 1), _Cell("4% BAFO", 2)],
        ],
        header_row_index=0,
    )


FULL_TEXT = (
    "Condor Logistics UK Ltd  Version 1 — Initial Rate Schedule ... "
    "Total Value £238,432 £230,930 £219,000 ... Surcharges sub-total £19,920 "
    "ANNUAL CONTRACT VALUE (EXCLUDING VAT) £238,432"
)


def test_doc_version_from_filename():
    assert _doc_version("", "documents/Quote/Freight_Condor_..._V1.pdf") == 1
    assert _doc_version("", "X_V3.pdf") == 3


def test_doc_version_from_masthead():
    assert _doc_version("Acme Co  Version 2 — Revised Rates", None) == 2


def test_recovers_v1_total_not_section_subtotal():
    got = _recover_contract_total(
        [_version_history_table()], FULL_TEXT,
        "documents/Quote/Freight_Condor_Logistics_UK_Ltd_V1.pdf",
    )
    assert got == 238432.0  # NOT 19920 (surcharges sub-total)


def test_recovers_correct_version_row():
    # A V3 document must take £219,000, not the V1 £238,432.
    ft = FULL_TEXT.replace("Version 1", "Version 3")
    got = _recover_contract_total([_version_history_table()], ft, "X_V3.pdf")
    assert got == 219000.0


def test_ignores_line_item_table():
    # An ordinary line-item table must never be treated as a totals table.
    line_tbl = _Table(rows=[
        [_Cell("Description", 0), _Cell("Qty", 1), _Cell("Unit Price", 2), _Cell("Total Value", 3)],
        [_Cell("Widget", 0), _Cell("2", 1), _Cell("£10", 2), _Cell("£20", 3)],
    ])
    assert _recover_contract_total([line_tbl], "Widget £20", "x.pdf") is None


def test_ungrounded_value_rejected():
    # If the table value isn't present in the document text, reject it.
    assert _recover_contract_total([_version_history_table()], "no totals here", "X_V1.pdf") is None


def test_no_tables_returns_none():
    assert _recover_contract_total(None, FULL_TEXT, "x.pdf") is None
    assert _recover_contract_total([], FULL_TEXT, "x.pdf") is None
