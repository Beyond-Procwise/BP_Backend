"""The report worker: whatever the pipeline does, the job leaves 'running'."""
from __future__ import annotations

from src.services.rga import job_runner
from src.services.rga.models import Finding, FindingCode, Severity
from src.services.rga.pipeline import ReportRun
from src.services.rga.render import RenderedArtefact


import pytest


@pytest.fixture(autouse=True)
def _no_audit_rows(monkeypatch):
    """A crash now writes report.run_failed. Under PROCWISE_TEST_LIVE_DB=1 that
    would be a permanent row in the append-only bp_agent_actions -- so it is
    captured here, and the test about that event captures it itself."""
    monkeypatch.setattr(job_runner.audit, "emit", lambda *a, **k: None)


class FakeStore:
    def __init__(self, claimable=True):
        self.claimable = claimable
        self.calls = []
        self.job = {"job_id": "rpt-1", "report_type": "exec_procurement_summary",
                    "scope": {"period_start": "2026-01-01"}, "as_of": "2026-09-24",
                    "run_id": "FP-a", "requested_by": "buyer-1",
                    "entitlement": {"action": "report.generate", "shadowed": False}}

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


def test_the_run_executes_inside_the_job_s_audit_context():
    from src.services.rga import audit

    store, seen = FakeStore(), {}

    def generate(report_type, **k):
        seen.update(audit.current_context())
        return ReportRun(run_id="FP-a", report_type_id=report_type, artefact=_artefact(),
                         released=True, stage_reached="RELEASE")

    job_runner.run_job("rpt-1", store=store, generate=generate)
    assert seen == {"job_id": "rpt-1", "requested_by": "buyer-1",
                    "entitlement": {"action": "report.generate", "shadowed": False}}
    assert audit.current_context() == {}


def test_a_crash_closes_the_trail_with_a_failed_event(monkeypatch):
    from src.services.rga import audit

    events = []
    monkeypatch.setattr(job_runner.audit, "emit",
                        lambda action, **k: events.append((action, k, audit.current_context())))

    def generate(*a, **k):
        raise RuntimeError("database went away")

    job_runner.run_job("rpt-1", store=FakeStore(), generate=generate)
    assert len(events) == 1
    action, k, ctx = events[0]
    assert action == audit.RUN_FAILED
    assert k["run_id"] == "FP-a" and k["status"] == "failed"
    assert "database went away" in k["summary"]
    assert ctx["job_id"] == "rpt-1"


def _released_run(report_type, **k):
    return ReportRun(run_id="FP-a", report_type_id=report_type, artefact=_artefact(),
                     released=True, stage_reached="RELEASE")


def test_a_release_that_needs_sign_off_asks_for_it(monkeypatch):
    from src.services.rga import audit
    events = []
    monkeypatch.setattr(job_runner.audit, "emit",
                        lambda action, **k: events.append((action, k, audit.current_context())))
    monkeypatch.setattr(job_runner.signoff, "required", lambda report_type: True)
    job_runner.run_job("rpt-1", store=FakeStore(), generate=_released_run)
    assert [e[0] for e in events] == [audit.APPROVAL_REQUESTED]
    action, k, ctx = events[0]
    assert k["run_id"] == "FP-a" and ctx["job_id"] == "rpt-1" and ctx["requested_by"] == "buyer-1"


def test_a_release_that_needs_none_asks_for_nothing(monkeypatch):
    events = []
    monkeypatch.setattr(job_runner.audit, "emit", lambda action, **k: events.append(action))
    monkeypatch.setattr(job_runner.signoff, "required", lambda report_type: False)
    job_runner.run_job("rpt-1", store=FakeStore(), generate=_released_run)
    assert events == []


def test_a_released_run_stores_its_page_beside_the_deck():
    store = FakeStore()
    page = RenderedArtefact(content=b"<html>page</html>", media_type="text/html; charset=utf-8",
                            renderer="html", renderer_version="1", pack_id="FP-a",
                            pack_hash="h", style_version="s", ast_hash="a")

    def generate(report_type, **k):
        return ReportRun(run_id="FP-a", report_type_id=report_type, artefact=_artefact(),
                         page=page, released=True, stage_reached="RELEASE")

    job_runner.run_job("rpt-1", store=store, generate=generate)
    kind, k = [c for c in store.calls if c[0] == "released"][0]
    assert k["page"] == b"<html>page</html>"
    assert k["page_media_type"] == "text/html; charset=utf-8"


def test_a_release_stores_what_an_editor_needs():
    """Editing re-renders from the STORED Fact Pack and AST (2026-09-24), so the worker hands
    both over with the title when it releases."""
    from types import SimpleNamespace
    store = FakeStore()
    pack = SimpleNamespace(model_dump=lambda **k: {"pack_id": "FP-a", "facts": []})
    ast = SimpleNamespace(model_dump=lambda **k: {"sections": []})

    def generate(report_type, **k):
        return ReportRun(run_id="FP-a", report_type_id=report_type, artefact=_artefact(),
                         pack=pack, ast=ast, released=True, stage_reached="RELEASE")

    job_runner.run_job("rpt-1", store=store, generate=generate)
    kind, k = [c for c in store.calls if c[0] == "released"][0]
    assert k["fact_pack"] == {"pack_id": "FP-a", "facts": []}
    assert k["ast"] == {"sections": []}
    assert k["title"] == "Executive procurement summary"
