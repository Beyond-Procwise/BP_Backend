import os
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.backend_scheduler as backend_scheduler


class DummyEmailWatcher:
    def __init__(self, *_, **__):
        self.started = False
        self.notifications: list[str] = []

    def start(self):
        self.started = True

    def stop(self, timeout: float = 5.0):  # pragma: no cover - simple stub
        self.started = False

    def notify_workflow(self, workflow_id: str):
        self.notifications.append(workflow_id)


class DummyTrainingEndpoint:
    def __init__(self):
        self.dispatched = []

    def dispatch(self, *, force=True, limit=None):
        self.dispatched.append({"force": force, "limit": limit})
        return {"training_jobs": [], "relationship_jobs": []}

    def configure_capture(self, enable):  # pragma: no cover - not used in these tests
        self.capture_state = enable

    def get_service(self):  # pragma: no cover - helper to satisfy interface
        return Mock()


def _prepare_scheduler(monkeypatch, nick, endpoint=None):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_register_default_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )
    monkeypatch.setattr(backend_scheduler, "EmailWatcherService", DummyEmailWatcher)
    if endpoint is not None:
        monkeypatch.setattr(
            backend_scheduler.BackendScheduler,
            "_resolve_training_endpoint",
            lambda self: endpoint,
        )
    return backend_scheduler.BackendScheduler(nick, training_endpoint=endpoint)


def _spawn_scheduler(monkeypatch, settings):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )
    monkeypatch.setattr(backend_scheduler, "EmailWatcherService", DummyEmailWatcher)
    nick = SimpleNamespace(settings=settings)
    return backend_scheduler.BackendScheduler(nick)


def test_submit_once_executes_and_removes_job(monkeypatch):
    scheduler = _prepare_scheduler(monkeypatch, SimpleNamespace())

    executed = []

    scheduler.submit_once("once", lambda: executed.append("ran"))

    job = scheduler._jobs["once"]
    scheduler._execute_job(job)

    assert executed == ["ran"]
    assert "once" not in scheduler._jobs
    assert isinstance(scheduler._email_watcher_service, DummyEmailWatcher)
    assert scheduler._email_watcher_service.started is True

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_run_model_training_dispatches_via_endpoint(monkeypatch):
    endpoint = DummyTrainingEndpoint()
    scheduler = _prepare_scheduler(
        monkeypatch,
        SimpleNamespace(),
        endpoint=endpoint,
    )

    scheduler._run_model_training()

    assert endpoint.dispatched == [{"force": False, "limit": None}]

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_ensure_updates_training_endpoint_reference(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_register_default_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )

    first_endpoint = DummyTrainingEndpoint()
    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(), training_endpoint=first_endpoint
    )
    assert scheduler._training_endpoint is first_endpoint

    second_endpoint = DummyTrainingEndpoint()
    backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(), training_endpoint=second_endpoint
    )
    assert scheduler._training_endpoint is second_endpoint

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_notify_email_dispatch_wakes_watcher(monkeypatch):
    scheduler = _prepare_scheduler(monkeypatch, SimpleNamespace())

    assert isinstance(scheduler._email_watcher_service, DummyEmailWatcher)
    assert scheduler._email_watcher_service.started is True

    scheduler.notify_email_dispatch("wf-demo")

    assert scheduler._email_watcher_service.notifications == ["wf-demo"]

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_resolve_training_endpoint_creates_instance(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_register_default_jobs",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )

    scheduler = backend_scheduler.BackendScheduler(SimpleNamespace())
    endpoint = scheduler._resolve_training_endpoint()

    from services.model_training_endpoint import ModelTrainingEndpoint

    assert isinstance(endpoint, ModelTrainingEndpoint)

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_training_scheduler_skips_job_when_setting_missing(monkeypatch):
    scheduler = _spawn_scheduler(monkeypatch, SimpleNamespace())

    job_name = backend_scheduler.BackendScheduler.TRAINING_JOB_NAME
    assert job_name not in scheduler._jobs

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_training_scheduler_registers_job_when_enabled(monkeypatch):
    scheduler = _spawn_scheduler(
        monkeypatch, SimpleNamespace(enable_training_scheduler=True)
    )

    job_name = backend_scheduler.BackendScheduler.TRAINING_JOB_NAME
    assert job_name in scheduler._jobs
    assert scheduler._jobs[job_name].interval == timedelta(hours=6)

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_training_scheduler_reconfigures_on_ensure(monkeypatch):
    backend_scheduler.BackendScheduler._instance = None
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "_init_relationship_scheduler",
        lambda self: None,
    )
    monkeypatch.setattr(
        backend_scheduler.BackendScheduler,
        "start",
        lambda self: None,
    )
    monkeypatch.setattr(backend_scheduler, "EmailWatcherService", DummyEmailWatcher)

    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(settings=SimpleNamespace())
    )
    job_name = backend_scheduler.BackendScheduler.TRAINING_JOB_NAME
    assert job_name not in scheduler._jobs

    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(settings=SimpleNamespace(enable_training_scheduler=True))
    )
    assert job_name in scheduler._jobs

    scheduler = backend_scheduler.BackendScheduler.ensure(
        SimpleNamespace(settings=SimpleNamespace(enable_training_scheduler=False))
    )
    assert job_name not in scheduler._jobs

    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None



# --- job lanes -------------------------------------------------------------
# The scheduler used to execute every due job inline, one after another, on the
# single polling thread. A slow job therefore delayed every job registered after
# it AND stalled the poll loop itself, so nothing else was even evaluated as due.
# Observed live 2026-08-01: deal-assignment ran 18:15:22 -> 18:19:42 doing no work
# at all (every counter zero) and the analysis sweep, registered directly after it,
# could not start until 18:19:42.
#
# Lanes fix that WITHOUT weakening the mutual exclusion the old design implied:
# jobs sharing a lane still never overlap, and the default lane is shared, so a job
# has to opt in to concurrency.

def _job_scheduler(monkeypatch):
    return _prepare_scheduler(monkeypatch, SimpleNamespace())


def test_a_slow_job_does_not_delay_a_job_in_another_lane(monkeypatch):
    import threading
    from datetime import datetime, timezone

    scheduler = _job_scheduler(monkeypatch)
    hold, slow_started, fast_ran = threading.Event(), threading.Event(), threading.Event()

    def slow():
        slow_started.set()
        hold.wait(5)

    scheduler.register_job("slow", slow, interval=timedelta(minutes=15))
    scheduler.register_job("fast", fast_ran.set, interval=timedelta(minutes=15),
                           lane="analysis")

    scheduler._dispatch_due(datetime.now(timezone.utc))

    assert slow_started.wait(5), "the slow job should have started"
    # THE POINT: this must not wait on the slow job in the other lane.
    assert fast_ran.wait(5), "a job in its own lane must not queue behind a slow one"
    hold.set()
    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_jobs_sharing_a_lane_never_run_at_the_same_time(monkeypatch):
    """The guarantee the old sequential loop gave implicitly, kept explicitly.
    trgt-promotion and deal-assignment both write the _trgt tables and must not
    overlap, so the default lane stays strictly one-at-a-time."""
    import threading
    from datetime import datetime, timezone

    scheduler = _job_scheduler(monkeypatch)
    hold, first_started, second_ran = threading.Event(), threading.Event(), threading.Event()

    def first():
        first_started.set()
        hold.wait(5)

    scheduler.register_job("first", first, interval=timedelta(minutes=15))
    scheduler.register_job("second", second_ran.set, interval=timedelta(minutes=15))

    scheduler._dispatch_due(datetime.now(timezone.utc))
    assert first_started.wait(5)

    assert not second_ran.wait(0.3), "same-lane jobs must never overlap"
    hold.set()
    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_a_job_held_off_by_a_busy_lane_is_not_lost(monkeypatch):
    """Skipping is not dropping: the job stays due and runs on a later poll,
    and its next_run is NOT advanced as though it had run."""
    import threading
    from datetime import datetime, timezone

    scheduler = _job_scheduler(monkeypatch)
    hold, first_started, second_ran = threading.Event(), threading.Event(), threading.Event()

    def first():
        first_started.set()
        hold.wait(5)

    scheduler.register_job("first", first, interval=timedelta(minutes=15))
    scheduler.register_job("second", second_ran.set, interval=timedelta(minutes=15))
    second = scheduler._jobs["second"]
    due_at = second.next_run

    scheduler._dispatch_due(datetime.now(timezone.utc))
    assert first_started.wait(5)
    assert second.next_run == due_at, "a skipped job must stay due"

    hold.set()
    for _ in range(50):                       # let the lane drain
        if not scheduler._lane_busy("pipeline"):
            break
        threading.Event().wait(0.1)

    scheduler._dispatch_due(datetime.now(timezone.utc))
    assert second_ran.wait(5), "the held-off job must run on a later poll"
    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_the_analysis_sweep_gets_a_lane_of_its_own(monkeypatch):
    """bp_analysis* is written only by analysis_store, whose start() is idempotent
    (ON CONFLICT) and whose freeze() takes FOR UPDATE — it is already safe against
    the live session listener, so it is safe off the shared pipeline lane."""
    scheduler = _job_scheduler(monkeypatch)
    scheduler._register_analysis_sweep_job()

    sweep = scheduler._jobs[scheduler.ANALYSIS_SWEEP_JOB_NAME]
    scheduler.register_job("anything", lambda: None, interval=timedelta(minutes=15))

    assert sweep.lane != scheduler._jobs["anything"].lane
    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


def test_the_default_lane_is_shared_so_concurrency_is_opt_in(monkeypatch):
    """Regression guard: a job added later must not become concurrent with the
    data-pipeline jobs merely by being registered."""
    scheduler = _job_scheduler(monkeypatch)
    scheduler.register_job("a", lambda: None, interval=timedelta(minutes=15))
    scheduler.register_job("b", lambda: None, interval=timedelta(minutes=15))

    assert scheduler._jobs["a"].lane == scheduler._jobs["b"].lane
    scheduler.stop()
    backend_scheduler.BackendScheduler._instance = None


# The graph-resolution chain opens a database connection and a Neo4j driver, so
# a scheduler built with object.__new__ carries none of what the body needs: if
# either gate stops holding, these fail loudly instead of running the pass.
_CHANGED = {"forward_linked": 3}


def test_graph_resolution_does_not_run_when_no_deal_changed(monkeypatch):
    monkeypatch.setenv("GRAPH_RESOLUTION_ENABLED", "1")
    scheduler = object.__new__(backend_scheduler.BackendScheduler)

    assert scheduler._chain_graph_resolution({"forward_linked": 0}) is None


def test_graph_resolution_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("GRAPH_RESOLUTION_ENABLED", "0")
    scheduler = object.__new__(backend_scheduler.BackendScheduler)

    assert scheduler._chain_graph_resolution(_CHANGED) is None
