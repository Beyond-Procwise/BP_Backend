import pytest
from src.services.structural_extractor.parsing.dispatcher import detect_format


def test_detect_pdf_by_magic():
    assert detect_format(b"%PDF-1.4\n...", "inv.pdf") == "pdf"


def test_detect_docx_by_ext():
    assert detect_format(b"PK\x03\x04something", "inv.docx") == "docx"


def test_detect_xlsx_by_ext():
    assert detect_format(b"PK\x03\x04", "inv.xlsx") == "xlsx"


def test_detect_csv_by_ext():
    assert detect_format(b"a,b,c\n1,2,3\n", "inv.csv") == "csv"


def test_reject_unknown():
    with pytest.raises(ValueError, match="Cannot detect format"):
        detect_format(b"xxx", "inv.bin")
