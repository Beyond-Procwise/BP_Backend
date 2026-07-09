from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.api.routers.ws import _get_resolved_session


def _cursor_with(rows):
    cur = MagicMock()
    cur.fetchone.side_effect = rows
    cur.__enter__ = lambda s: s
    cur.__exit__ = lambda s, *a: False
    return cur


def test_payload_includes_doc_action_breakdown():
    # 3 fetchone() calls: (1) rollup row, (2) category/deal row, (3) doc_action row
    rollup = ("completed", 2, 2, 0, 0, None)          # action_status,total,target,discrep,failed,resolved_at
    extra = (["po"], ["Deal A"])                        # category, deal_name
    doc_actions = (1, 0, 0, 0, [{"file_path": "a.pdf", "doc_action": "duplicate"}])
    cur = _cursor_with([rollup, extra, doc_actions])
    conn = MagicMock()
    conn.cursor.return_value = cur

    @contextmanager
    def fake_get_conn():
        yield conn

    with patch("services.db.get_conn", fake_get_conn):
        payload = _get_resolved_session("SESSION-1")

    assert payload["duplicate"] == 1
    assert payload["updated"] == 0
    assert payload["needs_review"] == 0
    assert payload["unsupported"] == 0
    assert payload["documents"] == [{"file_path": "a.pdf", "doc_action": "duplicate"}]
    assert payload["action_status"] == "completed"
