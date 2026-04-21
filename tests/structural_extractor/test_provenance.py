from unittest.mock import MagicMock

from src.services.structural_extractor.parsing.model import BBox
from src.services.structural_extractor.provenance import write_provenance
from src.services.structural_extractor.types import ExtractedValue


def test_write_provenance_executes_inserts():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    header = {
        "invoice_id": ExtractedValue(
            value="INV001", provenance="extracted",
            anchor_text="INV001", anchor_ref=BBox(1, 0, 0, 50, 20),
            source="structural", confidence=1.0, attempt=1,
        ),
        "due_date": ExtractedValue(
            value="2020-05-01", provenance="derived",
            derivation_trace={"rule_id": "due_date_default", "inputs": {"invoice_date": "2020-04-01"}},
            source="derivation_registry", confidence=1.0, attempt=1,
        ),
    }
    count = write_provenance(mock_conn, "bp_invoice", "INV001", header)
    assert count == 2
    assert mock_cur.execute.call_count == 2


def test_write_provenance_skips_none_values():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    header = {
        "invoice_id": ExtractedValue(value=None, provenance="extracted"),
        "supplier_name": ExtractedValue(
            value="Acme", provenance="extracted", confidence=1.0, attempt=1,
        ),
    }
    count = write_provenance(mock_conn, "bp_invoice", "INV001", header)
    assert count == 1
    assert mock_cur.execute.call_count == 1


def test_write_provenance_skips_none_entries():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    header = {
        "invoice_id": None,
        "supplier_name": ExtractedValue(
            value="Acme", provenance="extracted", confidence=1.0, attempt=1,
        ),
    }
    count = write_provenance(mock_conn, "bp_invoice", "INV001", header)
    assert count == 1
