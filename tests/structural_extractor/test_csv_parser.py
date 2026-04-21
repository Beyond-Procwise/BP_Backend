from src.services.structural_extractor.parsing.csv_parser import parse_csv
from src.services.structural_extractor.parsing.model import ColumnRef


def test_parse_csv_with_header():
    data = b"invoice_id,amount,currency\nINV-001,100.0,GBP\nINV-002,200.0,USD\n"
    doc = parse_csv(data, "test.csv")
    assert doc.source_format == "csv"
    inv_token = next(t for t in doc.tokens if t.text == "INV-001")
    assert isinstance(inv_token.anchor, ColumnRef)
    assert inv_token.anchor.column_name == "invoice_id"


def test_parse_csv_without_header():
    data = b"INV-001,100.0,GBP\nINV-002,200.0,USD\n"
    doc = parse_csv(data, "test.csv")
    assert len(doc.tables) == 1
