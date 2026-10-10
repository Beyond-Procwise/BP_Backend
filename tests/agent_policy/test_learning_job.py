"""The hourly learning scan's scheduler registration, its env toggle and its lane."""
import pytest

from services.backend_scheduler import BackendScheduler


class _ScanStub:
    AGENT_POLICY_LEARNING_SCAN_JOB_NAME = BackendScheduler.AGENT_POLICY_LEARNING_SCAN_JOB_NAME
    AGENT_POLICY_LEARNING_SCAN_LANE = BackendScheduler.AGENT_POLICY_LEARNING_SCAN_LANE
    _register_agent_policy_learning_scan_job = BackendScheduler._register_agent_policy_learning_scan_job
    _run_agent_policy_learning_scan = BackendScheduler._run_agent_policy_learning_scan

    def __init__(self):
        self._jobs = {}

    def register_job(self, name, runner, interval, initial_delay=None, lane="pipeline", **_):
        self._jobs[name] = (runner, interval, initial_delay, lane)


@pytest.mark.parametrize("value,registered", [(None, True), ("on", True), ("OFF", False), ("off", False),
                                              ("anything", True)])
def test_learning_scan_toggle(monkeypatch, value, registered):
    if value is None:
        monkeypatch.delenv("AGENT_POLICY_LEARNING_SCAN", raising=False)
    else:
        monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", value)
    s = _ScanStub()
    s._register_agent_policy_learning_scan_job()
    assert (s.AGENT_POLICY_LEARNING_SCAN_JOB_NAME in s._jobs) is registered
    if registered:
        _runner, interval, delay, _lane = s._jobs[s.AGENT_POLICY_LEARNING_SCAN_JOB_NAME]
        assert interval.total_seconds() == 3600 and delay.total_seconds() == 600


def test_learning_scan_has_its_own_lane(monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_LEARNING_SCAN", raising=False)
    s = _ScanStub()
    s._register_agent_policy_learning_scan_job()
    lane = s._jobs[s.AGENT_POLICY_LEARNING_SCAN_JOB_NAME][3]
    assert lane == "agent-policy-learning"
    assert lane not in ("pipeline", BackendScheduler.AGENT_POLICY_CONFLICT_SCAN_LANE)


def test_learning_scan_is_registered_at_startup():
    import inspect
    src = inspect.getsource(BackendScheduler)
    assert "self._register_agent_policy_learning_scan_job()" in src


def test_learning_scan_calls_scan_and_never_raises(monkeypatch):
    from services.agent_policy import learning
    import services.db as db

    class _Ctx:
        def __enter__(self): return object()
        def __exit__(self, *a): return False
    monkeypatch.setattr(db, "get_conn", lambda: _Ctx())
    calls = []
    monkeypatch.setattr(learning, "scan", lambda conn, now: calls.append(now) or {"policies": 1, "raised": 1})
    _ScanStub()._run_agent_policy_learning_scan()
    assert len(calls) == 1 and calls[0].tzinfo is not None
    monkeypatch.setattr(learning, "scan", lambda conn, now: 1 / 0)
    _ScanStub()._run_agent_policy_learning_scan()


def test_learning_scan_says_so_when_a_suggestion_row_could_not_be_read(monkeypatch, caplog):
    import logging

    from services.agent_policy import learning
    import services.db as db

    class _Ctx:
        def __enter__(self): return object()
        def __exit__(self, *a): return False
    monkeypatch.setattr(db, "get_conn", lambda: _Ctx())
    quiet = {k: 0 for k in learning.COUNTERS}
    assert "unreadable" in quiet and "skipped" in quiet
    for counts, logged in (({**quiet, "policies": 4, "updated": 2}, False), ({**quiet, "policies": 4, "unreadable": 1}, True),
                           ({**quiet, "policies": 4, "skipped": 1}, True)):
        caplog.clear()
        monkeypatch.setattr(learning, "scan", lambda conn, now, counts=counts: counts)
        with caplog.at_level(logging.INFO, logger="services.backend_scheduler"):
            _ScanStub()._run_agent_policy_learning_scan()
        assert bool([r for r in caplog.records if "agent policy learning scan" in r.getMessage()]) is logged
