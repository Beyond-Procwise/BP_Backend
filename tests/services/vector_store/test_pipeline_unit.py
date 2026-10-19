"""Outbox SQL discipline and the extraction hook, without a database."""
import json
import subprocess
import sys

import pytest

from src.services.vector_store import outbox


class Cur:
    def __init__(self, rows=None, rowcount=1):
        self.executed, self.rows, self.rowcount = [], rows or [], rowcount
        self.description = [("job_id",), ("tenant_id",), ("collection",), ("source_id",), ("action",), ("profile",), ("attempts",), ("max_attempts",)]

    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return (1,) if self.rows else None


def test_claiming_uses_skip_locked_so_two_workers_never_take_the_same_job():
    cur = Cur()
    outbox.claim(cur, 5, 300)
    sql = cur.executed[0][0].lower()
    assert "for update skip locked" in sql and "limit %s" in sql and "next_attempt_at <= now()" in sql
    assert "processing" in sql and "interval" in sql                     # stale processing jobs are reclaimable


def test_a_failure_backs_off_exponentially_and_stops_at_max_attempts():
    cur = Cur()
    job = outbox.Job(1, "t", "c", "s", "upsert", None, attempts=1, max_attempts=5)
    outbox.fail(cur, job, "boom")
    sql, params = cur.executed[0]
    assert "status = %s" in sql and params[0] == "pending" and params[1] == 60                 # 30s * 2**attempts
    outbox.fail(cur, outbox.Job(1, "t", "c", "s", "upsert", None, attempts=5, max_attempts=5), "boom")
    assert cur.executed[1][1][0] == "failed"


def test_enqueue_requires_a_tenant_and_a_valid_action():
    with pytest.raises(ValueError):
        outbox.enqueue(Cur(), "", "c", "s")
    with pytest.raises(ValueError):
        outbox.enqueue(Cur(), "t", "c", "s", action="explode")


def test_the_rebuild_command_has_a_help_screen():
    out = subprocess.run([sys.executable, "-m", "src.services.vector_store.rebuild", "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--tenant" in out.stdout and "--profile" in out.stdout and "--no-clear" in out.stdout


# --- the extraction hook: same transaction, never blocks extraction -------------------------------------------------------------

class TxnCur:
    def __init__(self, fail_enqueue=False):
        self.executed, self.fail_enqueue = [], fail_enqueue

    def execute(self, sql, params=()):
        flat = " ".join(sql.split())
        self.executed.append(flat)
        if self.fail_enqueue and "bp_embedding_job" in flat:
            raise RuntimeError('relation "proc.bp_embedding_job" does not exist')

    def fetchone(self):
        return (7,)

    rowcount = 1


class TxnConn:
    def __init__(self, cur):
        self._cur, self.committed, self.rolled_back, self.autocommit = cur, False, False, True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _write(monkeypatch, cur, flag):
    from src.services.extraction import persistence
    conn = TxnConn(cur)
    monkeypatch.setattr(persistence, "get_conn", lambda: conn)
    if flag is None:
        monkeypatch.delenv("VECTOR_OUTBOX", raising=False)
    else:
        monkeypatch.setenv("VECTOR_OUTBOX", flag)
    raw_id = persistence.write_raw(doc_type="invoice", file_path="x.pdf", process_monitor_id=None, trace_id="00000000-0000-0000-0000-000000000001",
                                   pipeline_version="t", columns={"invoice_id": "INV-1"}, parser_snapshot={"full_text": "some text"},
                                   promotion_status="pending")
    return raw_id, conn


def test_write_raw_enqueues_in_the_same_transaction_when_the_outbox_is_on(monkeypatch):
    cur = TxnCur()
    raw_id, conn = _write(monkeypatch, cur, "1")
    assert raw_id == 7 and conn.committed and not conn.rolled_back
    assert any("insert into proc.bp_embedding_job" in s.lower() for s in cur.executed)
    order = [i for i, s in enumerate(cur.executed) if "insert into proc.bp_invoice_raw" in s.lower() or "bp_embedding_job" in s.lower()]
    assert len(order) == 2 and order[0] < order[1]                         # the raw row first, then its job, before the single commit


def test_write_raw_does_nothing_extra_when_the_outbox_is_off(monkeypatch):
    cur = TxnCur()
    _write(monkeypatch, cur, None)
    assert not any("bp_embedding_job" in s for s in cur.executed)


def test_a_missing_outbox_never_blocks_extraction(monkeypatch):
    cur = TxnCur(fail_enqueue=True)
    raw_id, conn = _write(monkeypatch, cur, "1")
    assert raw_id == 7 and conn.committed and not conn.rolled_back
    assert any(s.lower().startswith("savepoint") for s in cur.executed) and any("rollback to savepoint" in s.lower() for s in cur.executed)


def test_no_job_is_queued_for_a_document_with_no_text_or_no_key(monkeypatch):
    from src.services.extraction import persistence
    cur = TxnCur()
    conn = TxnConn(cur)
    monkeypatch.setattr(persistence, "get_conn", lambda: conn)
    monkeypatch.setenv("VECTOR_OUTBOX", "1")
    persistence.write_raw(doc_type="invoice", file_path="x", process_monitor_id=None, trace_id="00000000-0000-0000-0000-000000000001",
                          pipeline_version="t", columns={"invoice_id": "INV-1"}, parser_snapshot={"full_text": ""}, promotion_status="pending")
    persistence.write_raw(doc_type="invoice", file_path="x", process_monitor_id=None, trace_id="00000000-0000-0000-0000-000000000001",
                          pipeline_version="t", columns={}, parser_snapshot={"full_text": "text"}, promotion_status="pending")
    assert not any("bp_embedding_job" in s for s in cur.executed)
