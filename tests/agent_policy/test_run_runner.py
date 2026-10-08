"""Runner with a fake store: failure becomes a one-line failed run; the heartbeat stops."""
import contextlib
import logging
import threading
import time

from services.agent_policy import run_runner


class FakeStore:
    OWNER = "me"

    def __init__(self, claimable=True):
        self.claimable, self.beats, self.finished, self.items = claimable, 0, None, []
        self.finish_result = True
        self.queued_beats = []

    def claim(self, conn, run_id, owner):
        return self.claimable

    def get(self, conn, run_id, after_seq=0):
        return {"run_id": run_id}

    def beat(self, conn, run_id, owner):
        self.beats += 1

    def beat_queued(self, conn, owner):
        self.queued_beats.append(owner)

    def append_item(self, conn, run_id, **kw):
        self.items.append(kw)
        return len(self.items)

    def finish(self, conn, run_id, status, *, counts, error=None):
        self.finished = (status, counts, error)
        return self.finish_result


@contextlib.contextmanager
def _conn():
    yield object()


def _heartbeat_threads():
    return [t for t in threading.enumerate() if t.name == "policy-extraction-heartbeat"]


def test_work_raising_finishes_failed_with_one_line_error():
    store = FakeStore()

    def work(conn, run, emit):
        raise ValueError("boom\nsecond   line")

    run_runner.run(1, work, store=store, conn_factory=_conn)
    status, _, error = store.finished
    assert status == "failed" and error == "boom second line" and "\n" not in error


def test_success_finishes_done_with_counts_and_emit_appends():
    store = FakeStore()

    def work(conn, run, emit):
        assert emit("note", {"a": 1}, reference="r") == 1
        return {"policies": 1}

    run_runner.run(2, work, store=store, conn_factory=_conn)
    assert store.finished == ("done", {"policies": 1}, None)
    assert store.items == [{"kind": "note", "payload": {"a": 1}, "reference": "r"}]


def test_heartbeat_beats_while_working_and_stops_after():
    store = FakeStore()

    def work(conn, run, emit):
        time.sleep(0.35)
        return {}

    run_runner.run(3, work, store=store, conn_factory=_conn, interval=0.05)
    assert store.beats >= 2
    assert not _heartbeat_threads()
    seen = store.beats
    time.sleep(0.2)
    assert store.beats == seen


def test_heartbeat_stops_when_work_fails():
    store = FakeStore()

    def work(conn, run, emit):
        raise RuntimeError("x")

    run_runner.run(4, work, store=store, conn_factory=_conn, interval=0.05)
    assert not _heartbeat_threads()


def test_unclaimable_run_is_left_alone_and_never_raises():
    store = FakeStore(claimable=False)
    called = []
    run_runner.run(5, lambda *a: called.append(1), store=store, conn_factory=_conn)
    assert not called and store.finished is None


def test_finish_false_logs_a_warning(caplog):
    store = FakeStore()
    store.finish_result = False
    with caplog.at_level(logging.WARNING):
        run_runner.run(6, lambda *a: {"x": 1}, store=store, conn_factory=_conn)
    assert any("no longer active" in r.message for r in caplog.records)


def test_finish_true_logs_no_warning(caplog):
    store = FakeStore()
    with caplog.at_level(logging.WARNING):
        run_runner.run(7, lambda *a: {}, store=store, conn_factory=_conn)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_submit_failure_marks_run_failed_and_does_not_raise():
    store = FakeStore()

    class Dead:
        def submit(self, *a):
            raise RuntimeError("executor is shut down")

    run_runner.submit(8, lambda *a: {}, executor=Dead(), store=store, conn_factory=_conn)
    status, _, error = store.finished
    assert status == "failed" and "could not be queued" in error


def test_submit_stamps_this_processs_queued_runs_first():
    store = FakeStore()
    order = []

    class Exec:
        def submit(self, *a):
            order.append(("submitted", list(store.queued_beats)))

    run_runner.submit(9, lambda *a: {}, executor=Exec(), store=store, conn_factory=_conn)
    assert order == [("submitted", ["me"])]


def test_the_heartbeat_also_vouches_for_the_queued_runs():
    store = FakeStore()

    def work(conn, run, emit):
        time.sleep(0.3)
        return {}

    run_runner.run(10, work, store=store, conn_factory=_conn, interval=0.05)
    assert len(store.queued_beats) >= 2 and set(store.queued_beats) == {"me"}


def test_a_failed_stamp_still_queues_the_run():
    store = FakeStore()

    def boom(conn, owner):
        raise RuntimeError("db down")
    store.beat_queued = boom
    queued = []

    class Exec:
        def submit(self, *a):
            queued.append(a[1])

    run_runner.submit(11, lambda *a: {}, executor=Exec(), store=store, conn_factory=_conn)
    assert queued == [11] and store.finished is None
