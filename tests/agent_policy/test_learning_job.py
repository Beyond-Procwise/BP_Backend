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


# OFF unless switched on: on / 1 / true (any case, spaces trimmed). Unset, and every other value, is off.
@pytest.mark.parametrize("value,registered", [
    (None, False), ("", False), ("off", False), ("OFF", False), ("0", False), ("no", False), ("false", False),
    ("anything", False), ("yes", False), ("on", True), ("ON", True), (" On ", True), ("1", True), ("true", True),
    ("True", True), (" TRUE\n", True)])
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


@pytest.mark.parametrize("value", [None, "off", "garbage"])
def test_learning_scan_says_once_that_it_is_off_and_how_to_turn_it_on(monkeypatch, caplog, value):
    import logging
    if value is None:
        monkeypatch.delenv("AGENT_POLICY_LEARNING_SCAN", raising=False)
    else:
        monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", value)
    with caplog.at_level(logging.INFO, logger="services.backend_scheduler"):
        _ScanStub()._register_agent_policy_learning_scan_job()
    [line] = caplog.records
    assert line.levelno == logging.INFO
    assert "is off" in line.getMessage() and "AGENT_POLICY_LEARNING_SCAN=on" in line.getMessage()


def test_learning_scan_says_nothing_about_being_off_when_it_is_on(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", "on")
    with caplog.at_level(logging.INFO, logger="services.backend_scheduler"):
        _ScanStub()._register_agent_policy_learning_scan_job()
    assert caplog.records == []


def test_learning_scan_has_its_own_lane(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", "on")
    s = _ScanStub()
    s._register_agent_policy_learning_scan_job()
    lane = s._jobs[s.AGENT_POLICY_LEARNING_SCAN_JOB_NAME][3]
    assert lane == "agent-policy-learning"
    assert lane not in ("pipeline", BackendScheduler.AGENT_POLICY_CONFLICT_SCAN_LANE)


def _started(monkeypatch):
    """A real BackendScheduler as its start-up leaves its job table: the real _register_default_jobs
    and the real register_job run; every OTHER job's registration is a no-op, so nothing here
    depends on (or starts) any other job, a thread or a database."""
    import threading
    s = object.__new__(BackendScheduler)
    s._jobs, s._lock = {}, threading.Lock()
    for name in dir(BackendScheduler):
        if (name.startswith("_register_") or name == "_sync_training_job") and name not in (
                "_register_default_jobs", "_register_agent_policy_learning_scan_job"):
            monkeypatch.setattr(s, name, lambda *a, **k: None, raising=False)
    s._register_default_jobs()
    return s


def test_learning_scan_is_in_a_started_schedulers_job_table_when_switched_on(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", "on")
    s = _started(monkeypatch)
    assert list(s._jobs) == [BackendScheduler.AGENT_POLICY_LEARNING_SCAN_JOB_NAME]
    job = s._jobs[BackendScheduler.AGENT_POLICY_LEARNING_SCAN_JOB_NAME]
    assert job.runner == s._run_agent_policy_learning_scan and job.lane == "agent-policy-learning"
    assert job.interval.total_seconds() == 3600 and job.one_shot is False


def test_learning_scan_is_not_in_a_started_schedulers_job_table_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_LEARNING_SCAN", raising=False)
    assert _started(monkeypatch)._jobs == {}


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
