import os
from datetime import timedelta
from src.services.backend_scheduler import BackendScheduler


def test_deal_assignment_job_registers_when_enabled(monkeypatch):
    monkeypatch.setenv("DEAL_ASSIGNMENT_ENABLED", "1")
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}; s._lock = __import__("threading").Lock()
    s._register_deal_assignment_job()
    assert BackendScheduler.DEAL_ASSIGNMENT_JOB_NAME in s._jobs


def test_deal_assignment_job_disabled(monkeypatch):
    monkeypatch.setenv("DEAL_ASSIGNMENT_ENABLED", "0")
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}; s._lock = __import__("threading").Lock()
    s._register_deal_assignment_job()
    assert BackendScheduler.DEAL_ASSIGNMENT_JOB_NAME not in s._jobs
