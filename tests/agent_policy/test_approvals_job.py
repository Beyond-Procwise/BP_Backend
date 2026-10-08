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
