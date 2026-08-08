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


def test_mail_intake_sweep_skips_registration_when_table_absent(monkeypatch):
    """An unmigrated environment must not register the job at all — the whole
    point being that it should never get a chance to raise UndefinedTable
    every 60 seconds, forever."""
    monkeypatch.setattr(
        BackendScheduler, "_mail_intake_tables_present", staticmethod(lambda: False)
    )
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}; s._lock = __import__("threading").Lock()
    s._register_mail_intake_sweep_job()
    assert BackendScheduler.MAIL_INTAKE_SWEEP_JOB_NAME not in s._jobs


def test_mail_intake_sweep_registers_when_table_present(monkeypatch):
    monkeypatch.setattr(
        BackendScheduler, "_mail_intake_tables_present", staticmethod(lambda: True)
    )
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}; s._lock = __import__("threading").Lock()
    s._register_mail_intake_sweep_job()
    assert BackendScheduler.MAIL_INTAKE_SWEEP_JOB_NAME in s._jobs
