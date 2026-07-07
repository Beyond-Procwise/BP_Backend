"""T8: the extraction-feedback job registers behind its flag."""
import threading

from src.services.backend_scheduler import BackendScheduler


def _bare_scheduler():
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}
    s._lock = threading.RLock()
    return s


def test_job_registered_when_enabled(monkeypatch):
    monkeypatch.setenv("EXTRACTION_FEEDBACK_ENABLED", "1")
    s = _bare_scheduler()
    s._register_extraction_feedback_job()
    assert BackendScheduler.EXTRACTION_FEEDBACK_JOB_NAME in s._jobs


def test_job_absent_when_disabled(monkeypatch):
    monkeypatch.setenv("EXTRACTION_FEEDBACK_ENABLED", "0")
    s = _bare_scheduler()
    s._register_extraction_feedback_job()
    assert BackendScheduler.EXTRACTION_FEEDBACK_JOB_NAME not in s._jobs
