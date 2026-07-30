"""Re-processing a document must refresh its open findings, not duplicate them.

The Test Data_300726 re-extraction inserted identical line warnings up to 7x
(94 duplicated keys corpus-wide) because write_discrepancies was a bare
INSERT. The open-findings key is (doc_type, doc_pk_candidate, issue_type,
field_name); a partial unique index enforces it and the writer upserts.
"""
from contextlib import contextmanager

from src.services.extraction import persistence
from src.services.extraction.persistence import Discrepancy, write_discrepancies


class _Cur:
    def __init__(self):
        self.batches = []

    def executemany(self, sql, rows):
        self.sql = sql
        self.batches.append(list(rows))

    def execute(self, sql, params=()):
        self.last = (sql, params)

    def fetchone(self):
        return None


class _Conn:
    def __init__(self, cur): self._cur = cur
    def cursor(self): return self._cur
    def commit(self): pass
    def rollback(self): pass
    autocommit = False


def _write(monkeypatch, discrepancies):
    cur = _Cur()

    @contextmanager
    def fake_conn():
        yield _Conn(cur)

    monkeypatch.setattr(persistence, "get_conn", fake_conn)
    monkeypatch.setattr(persistence, "bulk_record", lambda *a, **k: None)
    n = write_discrepancies(doc_type="invoice", raw_id=7, source_file="f.xlsx",
                            doc_pk_candidate="INV-1", discrepancies=discrepancies)
    return n, cur


def _d(issue="line_missing_amount", field="line_items[3]", notes="n1"):
    return Discrepancy(field_name=field, issue_type=issue, severity="warning",
                       blocks_promotion=False, notes=notes)


def test_upsert_clause_targets_the_open_findings_key(monkeypatch):
    _, cur = _write(monkeypatch, [_d()])
    s = " ".join(cur.sql.lower().split())
    assert "on conflict" in s
    assert "do update" in s
    assert "coalesce(status, 'open') <> 'resolved'" in s.replace("status,'open'", "status, 'open'")


def test_same_finding_twice_in_one_batch_writes_once(monkeypatch):
    # Postgres rejects ON CONFLICT affecting a row twice in one statement, so
    # in-batch duplicates must collapse BEFORE the insert (latest wins).
    n, cur = _write(monkeypatch, [_d(notes="first"), _d(notes="second")])
    assert n == 1
    rows = cur.batches[0]
    assert len(rows) == 1
    assert rows[0][-1] == "second"


def test_distinct_findings_all_write(monkeypatch):
    n, cur = _write(monkeypatch, [_d(field="line_items[1]"), _d(field="line_items[2]"),
                                  _d(issue="line_missing_numbers", field="line_items[1]")])
    assert n == 3
    assert len(cur.batches[0]) == 3
