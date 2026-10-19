import contextlib
import logging
from datetime import timedelta

from src.services import backend_scheduler as bs


def test_the_job_is_registered_daily():
    assert bs.BackendScheduler.SALES_DETECTION_JOB_NAME == "sales_opportunity_detection"
    src = open(bs.__file__).read()
    assert src.index("self._register_sales_detection_job()") > src.index("self._register_sales_calibration_job()")
    sched = object.__new__(bs.BackendScheduler)
    sched._jobs = {}
    seen = {}
    sched.register_job = lambda name, fn, interval=None, **kw: (
        seen.update(name=name, fn=fn, interval=interval), sched._jobs.__setitem__(name, fn))
    sched._register_sales_detection_job()
    sched._register_sales_detection_job()  # idempotent
    assert seen["name"] == "sales_opportunity_detection"
    assert seen["interval"] == timedelta(days=1)
    assert seen["fn"] == sched._run_sales_detection


def test_a_failing_detector_is_logged_not_raised(monkeypatch, caplog):
    from src.services.sell_side import _db, detector

    def boom(conn, **kw):
        raise RuntimeError("detector down")

    monkeypatch.setattr(detector, "detect", boom)
    monkeypatch.setattr(_db, "transactional_conn", lambda: contextlib.nullcontext("CONN"))
    sched = object.__new__(bs.BackendScheduler)
    with caplog.at_level(logging.ERROR):
        assert sched._run_sales_detection() is None
    assert any("sales opportunity detection failed" in r.getMessage() for r in caplog.records)


def test_a_successful_run_calls_detect_once_and_logs_the_count(monkeypatch, caplog):
    from src.services.sell_side import _db, detector
    calls = []
    monkeypatch.setattr(detector, "detect", lambda conn, **kw: calls.append(conn) or [1, 2, 3])
    monkeypatch.setattr(_db, "transactional_conn", lambda: contextlib.nullcontext("CONN"))
    sched = object.__new__(bs.BackendScheduler)
    with caplog.at_level(logging.INFO):
        assert sched._run_sales_detection() is None
    assert calls == ["CONN"]
    assert any("sales opportunity detection: 3" in r.getMessage() for r in caplog.records)
