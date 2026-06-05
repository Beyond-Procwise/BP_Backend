import contextlib

import src.services.agent_actions as aa
from src.services.extraction import persistence
from src.services.extraction.persistence import Discrepancy


class _FakeCursor:
    """Records SQL but performs no real I/O."""

    def __init__(self):
        self.executed = []
        self.executemany_calls = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def executemany(self, sql, seq_of_params):
        self.executemany_calls.append((sql, list(seq_of_params)))


class _FakeConn:
    """Minimal stand-in for a psycopg connection — no DB I/O."""

    def __init__(self):
        self.autocommit = True
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self._cursor = _FakeCursor()

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True

    # write_discrepancies uses `with get_conn() as conn:`; the real get_conn is
    # a @contextmanager generator, so we patch it with a contextmanager-wrapped
    # generator (below) rather than making the conn itself a context manager.


def test_write_discrepancies_records_validation_actions(monkeypatch):
    captured = {}
    fake_conn = _FakeConn()

    @contextlib.contextmanager
    def fake_get_conn():
        # Mirror the real get_conn: yield the connection inside a context manager
        # so `with persistence.get_conn() as conn:` receives our fake — no real
        # Postgres connection is ever opened.
        yield fake_conn

    monkeypatch.setattr(persistence, "get_conn", fake_get_conn)

    def fake_bulk_record(rows, *, conn=None):
        captured["rows"] = list(rows)
        captured["conn_passed"] = conn is not None

    monkeypatch.setattr(persistence, "bulk_record", fake_bulk_record)

    discs = [
        Discrepancy(
            field_name="tax_amount", raw_value="10", expected_value="12",
            computed_value="12", issue_type="tax_mismatch", severity="critical",
            blocks_promotion=True, evidence_page=1, evidence_bbox=None,
            evidence_text="Tax 12", notes="",
        ),
    ]
    persistence.write_discrepancies(
        doc_type="invoice", raw_id=42, source_file="/tmp/x.pdf",
        doc_pk_candidate="INV-1", discrepancies=discs,
    )

    assert captured["conn_passed"] is True
    assert len(captured["rows"]) == 1
    row = captured["rows"][0]
    assert row["phase"] == aa.PHASE_VALIDATION
    assert row["action_type"] == "discrepancy"
    assert row["field_name"] == "tax_amount"
    assert row["doc_pk"] == "INV-1"
