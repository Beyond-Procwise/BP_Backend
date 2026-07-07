"""Spreadsheet (.xlsx/.csv) parsing → ParsedDocument, and end-to-end extraction."""
import os

import pytest

from src.services.extraction_v3.parsers.router import parse as route_parse
from src.services.extraction_v3.parsers.spreadsheet_backend import parse_csv, parse_xlsx

XLSX = "/home/muthu/Downloads/new/HR Quote v2.xlsx"


def test_csv_parses_to_markdown(tmp_path):
    f = tmp_path / "q.csv"
    f.write_text("Quote ID,Amount\nQTE-2026-01521,12000\nItem,Qty\nHR Programme,1\n")
    doc = parse_csv(f)
    assert doc.file_format == "spreadsheet"
    assert doc.parser_backend == "csv-reader"
    assert "QTE-2026-01521" in doc.full_text
    assert "12000" in doc.full_text
    assert "|" in doc.full_text  # rendered as a markdown pipe table
    # one page with a real table for the L2 line-item extractor
    assert doc.pages and doc.pages[0].tables


def test_router_dispatches_csv(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    doc = route_parse(f)
    assert doc.file_format == "spreadsheet" and "1" in doc.full_text


@pytest.mark.skipif(not os.path.exists(XLSX), reason="sample xlsx not present")
def test_xlsx_parses_real_quote():
    doc = parse_xlsx(XLSX)
    assert doc.file_format == "spreadsheet"
    assert doc.parser_backend == "openpyxl"
    assert len(doc.full_text) > 50
    assert doc.pages  # at least one sheet


@pytest.mark.skipif(not os.path.exists(XLSX), reason="sample xlsx not present")
def test_router_no_longer_raises_on_xlsx():
    # Previously raised ValueError("unsupported file format: .xlsx")
    doc = route_parse(XLSX)
    assert doc.file_format == "spreadsheet"
