"""proc.bp_egress_event — the queryable half of the egress record.

The audit's question was "can you reconstruct, for a given date, precisely what
left the boundary and under which policy version?" These assert the parts of
that answer this module can actually give, and are careful not to assert the
part it cannot: there is no policy model, so ``policy_version`` is NULL and a
test claiming otherwise would be coverage of a control that does not exist.
"""
from __future__ import annotations

import queue

import pytest

from src.services import egress_log


@pytest.fixture(autouse=True)
def _drain_queue():
    """Each test starts with an empty queue and no background writer.

    The worker is a daemon thread that flushes to a real database. Left running
    between tests it would race the assertions and write rows nobody asked for.
    """
    egress_log._queue = queue.Queue(maxsize=egress_log._MAX_QUEUE)
    egress_log._take_drops()
    yield
    egress_log._queue = queue.Queue(maxsize=egress_log._MAX_QUEUE)
    egress_log._take_drops()


# --------------------------------------------------------------------------
# The payload digest
# --------------------------------------------------------------------------

def test_a_body_is_hashed_not_stored():
    body = {"prompt": "Invoice from ELEANOR PRICE for GBP 4,120.00"}
    sha, n = egress_log.payload_digest(body)
    assert sha and len(sha) == 64
    assert n and n > 0
    assert "ELEANOR" not in sha, (
        "the digest must not be reversible to the payload — an audit log that "
        "copies the data it audits doubles the exposure it exists to measure"
    )


def test_the_same_payload_hashes_the_same_way_regardless_of_key_order():
    """Without sorted keys two identical requests produce two digests, and the
    hash proves nothing about whether the same payload was sent twice."""
    a, _ = egress_log.payload_digest({"model": "x", "prompt": "hello"})
    b, _ = egress_log.payload_digest({"prompt": "hello", "model": "x"})
    assert a == b


def test_different_payloads_hash_differently():
    a, _ = egress_log.payload_digest({"prompt": "pay GBP 100"})
    b, _ = egress_log.payload_digest({"prompt": "pay GBP 900"})
    assert a != b


def test_no_body_yields_no_digest():
    assert egress_log.payload_digest(None) == (None, None)


@pytest.mark.parametrize("body", [b"raw bytes", "a string", {"k": "v"}, [1, 2]])
def test_every_body_shape_is_hashable(body):
    sha, n = egress_log.payload_digest(body)
    assert sha and n


def test_an_unserialisable_body_is_not_fatal():
    class _Weird:
        def __repr__(self):
            raise RuntimeError("nope")

    # default=str is used, so this must not raise out of the digest helper.
    sha, n = egress_log.payload_digest({"x": _Weird()})
    assert (sha is None) or isinstance(sha, str)


# --------------------------------------------------------------------------
# Queueing
# --------------------------------------------------------------------------

def test_record_queues_without_touching_the_database(monkeypatch):
    """record() is called from the request path. It must not block on a DB."""
    monkeypatch.setattr(egress_log, "_ensure_worker", lambda: None)
    egress_log.record(purpose="fx_rates", destination="open.er-api.com",
                      method="GET", outcome="http_200")
    assert egress_log._queue.qsize() == 1


def test_record_never_raises(monkeypatch):
    monkeypatch.setattr(egress_log, "_ensure_worker",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        egress_log._ensure_worker()          # the stub really does raise
    # ...and record still must not, because it only observes the call.
    try:
        egress_log.record(purpose="fx_rates", destination="x",
                          method="GET", outcome="http_200")
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"record() raised {exc!r}")


def test_a_full_queue_drops_and_counts_rather_than_blocking(monkeypatch):
    """An audit log that silently loses entries is worse than one that admits
    to a gap: a reader can act on "43 dropped" and cannot act on absence."""
    monkeypatch.setattr(egress_log, "_ensure_worker", lambda: None)
    egress_log._queue = queue.Queue(maxsize=2)
    for _ in range(5):
        egress_log.record(purpose="fx_rates", destination="x",
                          method="GET", outcome="http_200")
    assert egress_log._queue.qsize() == 2
    assert egress_log._take_drops() == 3


def test_the_drop_becomes_its_own_row(monkeypatch):
    written = {}

    def _fake_conn():
        class _Cur:
            def executemany(self, sql, rows):
                written["rows"] = rows

        class _Conn:
            autocommit = True

            def cursor(self):
                return _Cur()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Conn()

    import src.services.db as db
    monkeypatch.setattr(db, "get_conn", _fake_conn)
    egress_log._count_drop()
    egress_log._count_drop()
    egress_log.flush([])

    rows = written.get("rows") or []
    assert rows, "the drop was not written"
    outcomes = [r[5] for r in rows]
    assert "events_dropped" in outcomes
    assert any("2 event(s) dropped" in str(r[6]) for r in rows)


# --------------------------------------------------------------------------
# The row shape
# --------------------------------------------------------------------------

def test_the_row_carries_no_policy_version_and_no_classification():
    """Both are empty because neither control exists. A row claiming a policy
    version it did not have would be worse than a NULL."""
    row = egress_log._row({
        "occurred_at": None, "purpose": "fx_rates", "destination": "x",
        "method": "GET", "outcome": "http_200", "detail": None,
        "payload_sha256": "abc", "payload_bytes": 3,
    })
    assert row[9] is None, "policy_version must be NULL until a policy exists"
    assert row[10] == [], "classification must be empty until a registry exists"
    assert row[1] == "default", "tenant_id is 'default' — there is no tenant dimension"


def test_a_flush_failure_is_swallowed(monkeypatch):
    """This observes outbound calls. An audit writer that can break the call it
    is watching is a worse problem than a gap in the audit."""
    import src.services.db as db

    def _boom():
        raise RuntimeError("database is down")

    monkeypatch.setattr(db, "get_conn", _boom)
    assert egress_log.flush([{
        "occurred_at": None, "purpose": "p", "destination": "d",
        "method": "GET", "outcome": "o", "detail": None,
        "payload_sha256": None, "payload_bytes": None,
    }]) == 0
