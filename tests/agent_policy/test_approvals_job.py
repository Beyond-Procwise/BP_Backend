"""The approval sweep's scheduler registration and its env toggle."""
import pytest

from services.backend_scheduler import BackendScheduler


class _Stub:
    AGENT_POLICY_APPROVALS_JOB_NAME = BackendScheduler.AGENT_POLICY_APPROVALS_JOB_NAME
    _register_agent_policy_approvals_job = BackendScheduler._register_agent_policy_approvals_job
    _run_agent_policy_approvals_sweep = BackendScheduler._run_agent_policy_approvals_sweep

    def __init__(self):
        self._jobs = {}

    def register_job(self, name, runner, interval, initial_delay=None, **_):
        self._jobs[name] = (runner, interval)


@pytest.mark.parametrize("value,registered", [(None, True), ("on", True), ("OFF", False), ("off", False),
                                              ("anything", True)])
def test_toggle(monkeypatch, value, registered):
    if value is None:
        monkeypatch.delenv("AGENT_POLICY_APPROVAL_SWEEP", raising=False)
    else:
        monkeypatch.setenv("AGENT_POLICY_APPROVAL_SWEEP", value)
    s = _Stub()
    s._register_agent_policy_approvals_job()
    assert (s.AGENT_POLICY_APPROVALS_JOB_NAME in s._jobs) is registered
    if registered:
        assert s._jobs[s.AGENT_POLICY_APPROVALS_JOB_NAME][1].total_seconds() == 60


def test_run_calls_sweep_and_never_raises(monkeypatch):
    from services.agent_policy import approvals
    calls = []
    monkeypatch.setattr(approvals, "sweep", lambda conn, now: calls.append(now) or {"escalated": 0})
    _Stub()._run_agent_policy_approvals_sweep()
    assert len(calls) == 1 and calls[0].tzinfo is not None
    monkeypatch.setattr(approvals, "sweep", lambda conn, now: 1 / 0)
    _Stub()._run_agent_policy_approvals_sweep()


def test_run_also_retries_lost_replays_and_never_raises(monkeypatch):
    from services.agent_policy import approvals, replay_retry
    import services.db as db

    class _Ctx:
        def __enter__(self): return object()
        def __exit__(self, *a): return False
    monkeypatch.setattr(db, "get_conn", lambda: _Ctx())
    monkeypatch.setattr(approvals, "sweep", lambda conn, now: {"escalated": 0})
    calls = []
    monkeypatch.setattr(replay_retry, "retry_lost_replays", lambda conn, now: calls.append(now) or {"retried": 1})
    _Stub()._run_agent_policy_approvals_sweep()
    assert len(calls) == 1 and calls[0].tzinfo is not None
    # a failing sweep does not stop the retry, and a failing retry does not raise
    monkeypatch.setattr(approvals, "sweep", lambda conn, now: 1 / 0)
    monkeypatch.setattr(replay_retry, "retry_lost_replays", lambda conn, now: calls.append(now) or 1 / 0)
    _Stub()._run_agent_policy_approvals_sweep()
    assert len(calls) == 2


# ---------------------------------------------------------------- conflict scan (stage 4)
class _ScanStub:
    AGENT_POLICY_CONFLICT_SCAN_JOB_NAME = BackendScheduler.AGENT_POLICY_CONFLICT_SCAN_JOB_NAME
    _register_agent_policy_conflict_scan_job = BackendScheduler._register_agent_policy_conflict_scan_job
    _run_agent_policy_conflict_scan = BackendScheduler._run_agent_policy_conflict_scan

    def __init__(self):
        self._jobs = {}

    def register_job(self, name, runner, interval, initial_delay=None, **_):
        self._jobs[name] = (runner, interval, initial_delay)


@pytest.mark.parametrize("value,registered", [(None, True), ("on", True), ("OFF", False), ("off", False),
                                              ("anything", True)])
def test_conflict_scan_toggle(monkeypatch, value, registered):
    if value is None:
        monkeypatch.delenv("AGENT_POLICY_CONFLICT_SCAN", raising=False)
    else:
        monkeypatch.setenv("AGENT_POLICY_CONFLICT_SCAN", value)
    s = _ScanStub()
    s._register_agent_policy_conflict_scan_job()
    assert (s.AGENT_POLICY_CONFLICT_SCAN_JOB_NAME in s._jobs) is registered
    if registered:
        _runner, interval, delay = s._jobs[s.AGENT_POLICY_CONFLICT_SCAN_JOB_NAME]
        assert interval.total_seconds() == 3600 and delay.total_seconds() == 300


def test_conflict_scan_is_registered_at_startup():
    import inspect
    src = inspect.getsource(BackendScheduler)
    assert "self._register_agent_policy_conflict_scan_job()" in src


def test_conflict_scan_calls_detect_all_and_never_raises(monkeypatch):
    from services.agent_policy import conflict_cases
    import services.db as db

    class _Ctx:
        def __enter__(self): return object()
        def __exit__(self, *a): return False
    monkeypatch.setattr(db, "get_conn", lambda: _Ctx())
    calls = []
    monkeypatch.setattr(conflict_cases, "detect_all",
                        lambda conn, now=None: calls.append(now) or {"pairs": 1, "raised": 1, "errors": 0})
    _ScanStub()._run_agent_policy_conflict_scan()
    assert len(calls) == 1 and calls[0].tzinfo is not None
    monkeypatch.setattr(conflict_cases, "detect_all", lambda conn, now=None: 1 / 0)
    _ScanStub()._run_agent_policy_conflict_scan()
