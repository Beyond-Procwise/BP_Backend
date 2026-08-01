"""T6: the analysis-sweep scheduler job. Covers _register_analysis_sweep_job
(name, enable flag, interval) and _run_analysis_sweep (calls
analysis_store.sweep, logs only when something happened).

Uses BackendScheduler.__new__() + a bare _jobs/_lock, same pattern as
tests/services/test_backend_scheduler.py and
tests/extraction_feedback/test_scheduler_registration.py, so nothing here
touches a real scheduler thread or database.

NOTE ON COLLECTION: this file is expected to hit the same PRE-EXISTING,
repo-wide collection error as those two sibling files (see GLOBAL
CONSTRAINTS: "~28 errors under parts of tests/services/ - missing unrelated
modules like services.email_watcher"). Root cause, confirmed by direct
investigation: tests/services/__init__.py exists (making tests/services a
real package) while tests/__init__.py does not, so pytest's import-mode
resolution inserts the repo's tests/ directory onto sys.path so it can import
this file as top-level package "services". That shadows the bare `import
services.email_watcher` (etc.) inside src/services/backend_scheduler.py's own
module-level imports - `services` resolves to tests/services instead of
src/services, and email_watcher.py isn't there.

This is not fixable from inside this file without doing something worse: a
sys.modules-surgery workaround was tried (popping the "services" entry and
re-inserting src/ ahead of tests/ on sys.path) and it corrupted pytest's own
module bookkeeping (KeyError during collection, since pytest itself needs
"services" to resolve to tests/services to import ITS OWN test modules under
this directory). That is a worse failure mode than a clean, honest collection
error, so it was discarded. The two existing sibling files
(tests/services/test_backend_scheduler.py,
tests/extraction_feedback/test_scheduler_registration.py) already exhibit the
identical pre-existing error family - this file is consistent with them, not
a new regression.
"""
import threading

from src.services.backend_scheduler import BackendScheduler


def _bare_scheduler():
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}
    s._lock = threading.Lock()
    return s


def test_job_name_constant():
    assert BackendScheduler.ANALYSIS_SWEEP_JOB_NAME == "analysis-sweep"


def test_job_registers_when_enabled(monkeypatch):
    monkeypatch.setenv("ANALYSIS_SWEEP_ENABLED", "1")
    s = _bare_scheduler()
    s._register_analysis_sweep_job()
    assert BackendScheduler.ANALYSIS_SWEEP_JOB_NAME in s._jobs


def test_job_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("ANALYSIS_SWEEP_ENABLED", "0")
    s = _bare_scheduler()
    s._register_analysis_sweep_job()
    assert BackendScheduler.ANALYSIS_SWEEP_JOB_NAME not in s._jobs


def test_job_defaults_to_a_fifteen_minute_interval(monkeypatch):
    monkeypatch.delenv("ANALYSIS_SWEEP_INTERVAL_MINUTES", raising=False)
    s = _bare_scheduler()
    s._register_analysis_sweep_job()
    job = s._jobs[BackendScheduler.ANALYSIS_SWEEP_JOB_NAME]
    from datetime import timedelta
    assert job.interval == timedelta(minutes=15)


def test_job_honours_the_interval_env_var(monkeypatch):
    monkeypatch.setenv("ANALYSIS_SWEEP_INTERVAL_MINUTES", "30")
    s = _bare_scheduler()
    s._register_analysis_sweep_job()
    job = s._jobs[BackendScheduler.ANALYSIS_SWEEP_JOB_NAME]
    from datetime import timedelta
    assert job.interval == timedelta(minutes=30)


def test_run_analysis_sweep_calls_analysis_store_sweep(monkeypatch):
    from src.services import analysis_store

    calls = []
    monkeypatch.setattr(analysis_store, "sweep",
                        lambda: calls.append(1) or {"created": 0, "frozen": 0, "failed": 0})
    s = _bare_scheduler()

    s._run_analysis_sweep()

    assert calls == [1]


def test_run_analysis_sweep_logs_only_when_something_happened(monkeypatch, caplog):
    """A quiet system (nothing created/frozen/failed) must stay quiet in the
    logs - a sweep that logged every run would be noise, not signal."""
    from src.services import analysis_store
    import logging

    s = _bare_scheduler()
    caplog.set_level(logging.INFO, logger="src.services.backend_scheduler")

    monkeypatch.setattr(analysis_store, "sweep",
                        lambda: {"created": 0, "frozen": 0, "failed": 0})
    caplog.clear()
    s._run_analysis_sweep()
    assert not any("analysis sweep:" in r.message for r in caplog.records)

    monkeypatch.setattr(analysis_store, "sweep",
                        lambda: {"created": 1, "frozen": 0, "failed": 0})
    caplog.clear()
    s._run_analysis_sweep()
    assert any("analysis sweep:" in r.message for r in caplog.records)


def test_run_analysis_sweep_does_not_raise_when_sweep_fails(monkeypatch):
    """The scheduler thread must survive a sweep exception."""
    from src.services import analysis_store

    def boom():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(analysis_store, "sweep", boom)
    s = _bare_scheduler()

    s._run_analysis_sweep()  # must not raise
