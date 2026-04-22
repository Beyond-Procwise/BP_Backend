import io
import openpyxl
from src.services.structural_extractor.parsing import parse


def test_dispatch_to_csv():
    doc = parse(b"a,b\n1,2\n", "t.csv")
    assert doc.source_format == "csv"


def test_dispatch_to_xlsx():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws["A1"] = "x"
    buf = io.BytesIO()
    wb.save(buf)
    doc = parse(buf.getvalue(), "t.xlsx")
    assert doc.source_format == "xlsx"


def test_dispatch_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        parse(b"binary", "x.bin")
