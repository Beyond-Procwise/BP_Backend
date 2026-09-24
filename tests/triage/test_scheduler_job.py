from datetime import timedelta


def test_triage_job_registers_once():
    from src.services.backend_scheduler import BackendScheduler
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}
    calls = []

    def register_job(name, runner, interval, initial_delay=None, **kw):
        calls.append((name, interval))
        s._jobs[name] = runner

    s.register_job = register_job
    s._register_triage_job()
    s._register_triage_job()
    assert calls == [("discrepancy-triage", timedelta(minutes=15))]


def test_triage_job_never_raises(monkeypatch):
    from src.services import backend_scheduler
    from src.services.triage import engine

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(engine, "run_changed", boom)
    s = backend_scheduler.BackendScheduler.__new__(backend_scheduler.BackendScheduler)
    s._run_triage_job()   # logs, does not raise
