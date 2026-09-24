"""The report worker: whatever the pipeline does, the job leaves 'running'."""
from __future__ import annotations

from src.services.rga import job_runner
from src.services.rga.models import Finding, FindingCode, Severity
from src.services.rga.pipeline import ReportRun
from src.services.rga.render import RenderedArtefact


class FakeStore:
    def __init__(self, claimable=True):
        self.claimable = claimable
        self.calls = []
        self.job = {"job_id": "rpt-1", "report_type": "exec_procurement_summary",
                    "scope": {"period_start": "2026-01-01"}, "as_of": "2026-09-24"}

    def claim(self, job_id):
        self.calls.append(("claim", job_id))
        return self.claimable

    def get(self, job_id):
        return dict(self.job)

    def finish_released(self, job_id, **k):
        self.calls.append(("released", k))

    def finish_blocked(self, job_id, **k):
        self.calls.append(("blocked", k))

    def finish_failed(self, job_id, error, **k):
        self.calls.append(("failed", error))


def _artefact():
    return RenderedArtefact(content=b"deck", media_type="application/pptx",
                            renderer="pptx", renderer_version="1", pack_id="FP-a",
                            pack_hash="h", style_version="s", ast_hash="a")


def test_a_released_run_stores_the_deck():
    store, seen = FakeStore(), {}

    def generate(report_type, **k):
        seen.update(k, report_type=report_type)
        return ReportRun(run_id="FP-a", report_type_id=report_type, artefact=_artefact(),
                         released=True, stage_reached="RELEASE")

    job_runner.run_job("rpt-1", store=store, generate=generate)
    kind, k = store.calls[-1]
    assert kind == "released"
    assert k["deck"] == b"deck" and k["run_id"] == "FP-a"
    assert k["filename"] == "exec_procurement_summary_FP-a.pptx"
    # The job's own as-of, not "today" at the moment the worker got round to it.
    assert seen["as_of"] == "2026-09-24" and seen["scope"] == {"period_start": "2026-01-01"}


def test_a_blocked_run_stores_only_the_blocking_reasons():
    store = FakeStore()
    run = ReportRun(
        run_id="FP-a", report_type_id="exec_procurement_summary", artefact=_artefact(),
        released=False, stage_reached="POST_CHECK", findings=[
            Finding(finding_id="B", code=FindingCode.REPORT_UNTRACED_FIGURE,
                    severity=Severity.HIGH, detail="untraced"),
            Finding(finding_id="N", code=FindingCode.MEASURE_UNAVAILABLE,
                    severity=Severity.MEDIUM, detail="a note", blocks_release=False)])
    job_runner.run_job("rpt-1", store=store, generate=lambda *a, **k: run)
    kind, k = store.calls[-1]
    assert kind == "blocked"
    assert "deck" not in k
    assert [b["finding_id"] for b in k["blocking"]] == ["B"]
    assert k["blocking"][0]["code"] == "REPORT_UNTRACED_FIGURE"


def test_a_crash_marks_the_job_failed_instead_of_leaving_it_running():
    store = FakeStore()

    def generate(*a, **k):
        raise RuntimeError("database went away")

    job_runner.run_job("rpt-1", store=store, generate=generate)
    kind, error = store.calls[-1]
    assert kind == "failed"
    assert "database went away" in error


def test_a_job_already_claimed_is_not_run_twice():
    store, ran = FakeStore(claimable=False), []
    job_runner.run_job("rpt-1", store=store, generate=lambda *a, **k: ran.append(1))
    assert ran == []
    assert store.calls == [("claim", "rpt-1")]


def test_the_worker_runs_one_report_at_a_time():
    assert job_runner._EXECUTOR._max_workers == 1


def test_submitting_starts_one_heartbeat_that_keeps_beating(monkeypatch):
    import threading

    beats = threading.Event()
    calls = []

    def beat():
        calls.append(1)
        if len(calls) >= 2:
            beats.set()

    monkeypatch.setattr(job_runner, "_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(job_runner._store, "beat", beat)
    monkeypatch.setattr(job_runner, "_heartbeat", None)
    monkeypatch.setattr(job_runner._EXECUTOR, "submit", lambda *a, **k: None)
    job_runner.submit("rpt-1")
    first = job_runner._heartbeat
    job_runner.submit("rpt-2")
    assert job_runner._heartbeat is first            # one thread, not one per job
    assert beats.wait(2), "the heartbeat never beat twice"
