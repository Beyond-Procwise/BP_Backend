from unittest.mock import MagicMock

from src.services.structural_extractor.pattern_store import (
    PatternStore,
    get_trust_level,
)


def test_trust_level_thresholds():
    assert get_trust_level(0) == "none"
    assert get_trust_level(1) == "learning"
    assert get_trust_level(2) == "learning"
    assert get_trust_level(3) == "trusted"
    assert get_trust_level(10) == "trusted"


def test_save_anchor_patterns_writes_sql():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = None  # cache miss -> INSERT
    ps = PatternStore(lambda: mock_conn)
    ps.save_pattern_anchors(
        file_type="pdf", doc_type="Invoice", supplier_name="Acme",
        layout_signature="abc123",
        anchors={"invoice_id": {"pdf": {"page": 1, "x0": 420}}},
    )
    assert mock_cur.execute.called


def test_get_anchor_patterns_returns_cached():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = ({"invoice_id": {"pdf": {"page": 1}}},)
    ps = PatternStore(lambda: mock_conn)
    anchors = ps.get_pattern_anchors("pdf", "Invoice", "Acme", "abc123")
    assert anchors["invoice_id"]["pdf"]["page"] == 1


def test_get_trust_returns_level_from_success_count():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = (5,)
    ps = PatternStore(lambda: mock_conn)
    assert ps.get_trust("pdf", "Invoice", "Acme", "abc123") == "trusted"


def test_get_trust_returns_none_when_no_row():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = None
    ps = PatternStore(lambda: mock_conn)
    assert ps.get_trust("pdf", "Invoice", "Acme", "abc123") == "none"
