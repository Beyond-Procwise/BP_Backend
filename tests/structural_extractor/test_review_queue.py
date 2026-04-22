from unittest.mock import MagicMock

from src.services.structural_extractor.review_queue import park_in_review_queue
from src.services.structural_extractor.types import ExtractionResult


def test_park_writes_queue_row_and_updates_status():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    result = ExtractionResult(
        header={}, line_items=[], parsed_text="...", unresolved_fields=["invoice_date"],
        attempts=10, process_monitor_id=42, doc_type="Invoice",
    )
    park_in_review_queue(mock_conn, result, file_path="x.pdf")
    # Should have INSERT + UPDATE (2 execute calls)
    assert mock_cur.execute.call_count == 2


def test_park_without_process_monitor_id_skips_update():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    result = ExtractionResult(
        header={}, line_items=[], parsed_text="...", unresolved_fields=["invoice_date"],
        attempts=10, process_monitor_id=None, doc_type="Invoice",
    )
    park_in_review_queue(mock_conn, result, file_path="x.pdf")
    # Only INSERT, no UPDATE
    assert mock_cur.execute.call_count == 1
