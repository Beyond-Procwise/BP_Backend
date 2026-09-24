"""/reports: the Report Generation Agent's front door.

A report takes the local model minutes, so the door files a job and answers at
once; the caller polls the job and collects the deck. The store and the worker
are replaced here -- what is under test is the door: who may open it, what they
must bring, and that a deck leaves only for a released job.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.routers import reports as rr
from src.services import actions, guardrail

CALLER = "sub-real-caller"
_ALLOWED = guardrail.Decision(allowed=True, reason="compute is a reversible class",
                              policy_name="RoleDefinitionPolicy")
_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
BODY = {"report_type": "exec_procurement_summary", "period_start": "2026-01-01",
        "period_end": "2026-03-31", "period_label": "2026 Q1", "currency": "GBP"}


class _P:
    subject = CALLER


class FakeStore:
    def __init__(self):
        self.jobs, self.decks, self.created = {}, {}, []

    def create(self, report_type, *, scope, as_of, requested_by, entitlement=None):
        self.created.append((report_type, scope, as_of, requested_by))
        self.entitlement = entitlement
        for job in self.jobs.values():
            if job["status"] in ("queued", "running") and job["scope"] == scope:
                return dict(job), False
        job = {"job_id": f"rpt-{len(self.jobs) + 1}", "report_type": report_type,
               "scope": scope, "as_of": as_of, "status": "queued",
               "requested_by": requested_by, "requested_at": "2026-09-24T09:00:00+00:00",
               "started_at": None, "finished_at": None, "run_id": None,
               "stage_reached": None, "blocking": None, "error": None}
        self.jobs[job["job_id"]] = job
        return dict(job), True

    def get(self, job_id):
        job = self.jobs.get(job_id)
        return dict(job) if job else None

    def deck(self, job_id):
        return self.decks.get(job_id)

    def needs_attention(self, limit):
        self.attention_limit = limit
        return [dict(j, rerun_status=None) for j in self.jobs.values()
                if j["status"] in ("blocked", "failed") and not j.get("dismissed_at")]

    def dismiss(self, job_id, *, by, reason):
        job = self.jobs.get(job_id)
        if not job or job["status"] not in ("blocked", "failed") or job.get("dismissed_at"):
            return False
        job.update(dismissed_at="2026-09-24T13:00:00+00:00", dismissed_by=by, dismiss_reason=reason)
        return True

    def recent(self, limit):
        self.recent_limit = limit
        jobs = sorted(self.jobs.values(), key=lambda j: j["job_id"], reverse=True)
        return [dict(j) for j in jobs[:limit]]


@pytest.fixture
def client(monkeypatch):
    gates, submitted, store = [], [], FakeStore()

    def _gate(action, principal, **k):
        gates.append((action, getattr(principal, "subject", None), k.get("context")))
        return _ALLOWED

    monkeypatch.setattr(rr, "gate", _gate)
    monkeypatch.setattr(rr, "job_store", store)
    monkeypatch.setattr(rr.job_runner, "submit", submitted.append)
    app = FastAPI()
    app.include_router(rr.router)
    app.dependency_overrides[rr.require_user] = lambda: _P()
    c = TestClient(app)
    c.gates, c.submitted, c.store = gates, submitted, store
    return c


def _release(store, job_id):
    store.jobs[job_id].update(status="released", run_id="FP-abc", stage_reached="RELEASE")
    store.decks[job_id] = (b"PK-deck-bytes", _PPTX, "exec_procurement_summary_FP-abc.pptx")


def test_report_generate_is_a_compute_action():
    # compute is reversible: building a report inside the tenant changes nothing.
    # Sending one out is report.export, a share, and stays refused by default.
    assert actions.action_class("report.generate") == "compute"
    assert actions.action_class("report.export") == "share"
    assert actions.action_class("report.read") == "read"


def test_types_lists_what_has_a_builder(client):
    r = client.get("/reports/types")
    assert r.status_code == 200
    assert "exec_procurement_summary" in r.json()["report_types"]


def test_generate_answers_at_once_with_a_queued_job(client):
    r = client.post("/reports/generate", json=BODY)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["job_id"] == "rpt-1" and body["status"] == "queued"
    assert body["already_requested"] is False
    assert client.submitted == ["rpt-1"]

    report_type, scope, as_of, requested_by = client.store.created[0]
    assert report_type == "exec_procurement_summary"
    assert scope == {"period_start": "2026-01-01", "period_end": "2026-03-31",
                     "period_label": "2026 Q1", "currency": "GBP"}
    assert requested_by == CALLER
    assert len(as_of) == 10                      # an ISO day, fixed at request time


def test_asking_again_while_it_runs_returns_the_same_job_and_starts_nothing(client):
    client.post("/reports/generate", json=BODY)
    r = client.post("/reports/generate", json=BODY)
    assert r.status_code == 202
    assert r.json()["job_id"] == "rpt-1" and r.json()["already_requested"] is True
    assert client.submitted == ["rpt-1"]         # the worker was handed it once


def test_the_gate_is_asked_first_and_by_the_token_holder(client):
    client.post("/reports/generate", json=BODY)
    action, subject, context = client.gates[0]
    assert (action, subject) == ("report.generate", CALLER)
    assert context["report_type"] == "exec_procurement_summary"


def test_a_refusal_files_no_job(client, monkeypatch):
    def _refuse(*a, **k):
        raise HTTPException(status_code=403, detail="refused by a rule")

    monkeypatch.setattr(rr, "gate", _refuse)
    r = client.post("/reports/generate", json=BODY)
    assert r.status_code == 403
    assert client.store.created == [] and client.submitted == []


def test_a_job_reports_its_status_and_never_its_bytes(client):
    client.post("/reports/generate", json=BODY)
    r = client.get("/reports/jobs/rpt-1")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert r.json()["deck_ready"] is False

    _release(client.store, "rpt-1")
    body = client.get("/reports/jobs/rpt-1").json()
    assert body["status"] == "released"
    assert body["deck_ready"] is True
    assert "deck" not in body


def test_a_blocked_job_says_why(client):
    client.post("/reports/generate", json=BODY)
    client.store.jobs["rpt-1"].update(
        status="blocked", run_id="FP-abc", stage_reached="POST_CHECK",
        blocking=[{"finding_id": "FP-abc-PC001", "code": "REPORT_UNTRACED_FIGURE",
                   "severity": "HIGH", "detail": "'£9,999' traces to no fact"}])
    body = client.get("/reports/jobs/rpt-1").json()
    assert body["status"] == "blocked"
    assert body["blocking"][0]["code"] == "report_untraced_figure"   # lower-case on the wire
    assert body["deck_ready"] is False


def test_a_released_deck_downloads(client):
    client.post("/reports/generate", json=BODY)
    _release(client.store, "rpt-1")
    client.gates.clear()
    r = client.get("/reports/jobs/rpt-1/deck")
    assert r.status_code == 200
    assert r.content == b"PK-deck-bytes"
    assert r.headers["content-type"] == _PPTX
    assert r.headers["x-report-run-id"] == "FP-abc"
    assert 'filename="exec_procurement_summary_FP-abc.pptx"' in r.headers["content-disposition"]
    assert [g[0] for g in client.gates] == ["report.read"]


@pytest.mark.parametrize("status", ["queued", "running", "blocked", "failed"])
def test_no_deck_leaves_for_a_job_that_was_not_released(client, status):
    client.post("/reports/generate", json=BODY)
    client.store.jobs["rpt-1"]["status"] = status
    # Even if bytes were somehow on file, the status decides.
    client.store.decks["rpt-1"] = (b"PK-deck-bytes", _PPTX, "x.pptx")
    r = client.get("/reports/jobs/rpt-1/deck")
    assert r.status_code == 409
    assert b"PK-deck-bytes" not in r.content
    assert status in r.json()["detail"]


def test_an_unknown_job_is_404(client):
    assert client.get("/reports/jobs/rpt-nope").status_code == 404
    assert client.get("/reports/jobs/rpt-nope/deck").status_code == 404


def test_an_unknown_report_type_is_refused_before_anything_runs(client):
    r = client.post("/reports/generate", json={**BODY, "report_type": "board_paper"})
    assert r.status_code == 404
    assert client.store.created == [] and client.gates == []


@pytest.mark.parametrize("patch", [
    {"period_start": "2026-04-01"},             # starts after it ends
    {"period_end": "not-a-date"},
    {"currency": "pounds"},
])
def test_a_malformed_period_is_refused(client, patch):
    r = client.post("/reports/generate", json={**BODY, **patch})
    assert r.status_code == 422
    assert client.store.created == []


def test_label_and_currency_have_defaults(client):
    body = {k: v for k, v in BODY.items() if k not in ("period_label", "currency")}
    assert client.post("/reports/generate", json=body).status_code == 202
    scope = client.store.created[0][1]
    assert scope["currency"] == "GBP"
    assert scope["period_label"] == "2026-01-01 to 2026-03-31"


def test_an_anonymous_caller_is_refused_when_auth_is_on(monkeypatch):
    # No override of require_user: the real dependency runs.
    import api.auth as auth

    class _V:
        def verify(self, token):
            raise auth.AuthError("bad")

    monkeypatch.setattr(auth, "_active_verifier", lambda: _V())
    app = FastAPI()
    app.include_router(rr.router)
    c = TestClient(app)
    assert c.post("/reports/generate", json=BODY).status_code == 401
    assert c.get("/reports/jobs/rpt-1/deck").status_code == 401


def test_replies_survive_the_output_safety_boundary(client, monkeypatch):
    """api/main.py scrubs every JSON reply, and an internal route in a field is
    withheld. Live 2026-09-24: deck_url came back "[withheld]" from procwise.
    So replies carry the job id and a flag, never a path; the caller builds it."""
    from src.services import output_safety as osafe

    monkeypatch.setattr(osafe, "_route_paths", {r.path for r in rr.router.routes})
    started = client.post("/reports/generate", json=BODY).json()
    _release(client.store, "rpt-1")
    status = client.get("/reports/jobs/rpt-1").json()
    # A blocked job too: its check codes (REPORT_UNTRACED_FIGURE) read as env-var
    # names to the boundary and came back "[withheld]" -- live, the Verified decks
    # "Why?" lost every reason but the composition one.
    client.post("/reports/generate", json={**BODY, "period_end": "2026-02-28"})
    client.store.jobs["rpt-2"].update(status="blocked", blocking=[
        {"finding_id": "F1", "code": "REPORT_UNTRACED_FIGURE", "severity": "HIGH", "detail": "x"}])
    blocked = client.get("/reports/jobs/rpt-2").json()
    listed = client.get("/reports/jobs").json()
    for reply in (started, status, blocked, listed):
        assert osafe.scrub_payload(reply, where="test") == reply
    assert status["deck_ready"] is True
    assert blocked["blocking"][0]["code"] == "report_untraced_figure"


def test_recent_jobs_are_listed_newest_first_without_bytes(client):
    client.post("/reports/generate", json=BODY)
    client.post("/reports/generate", json={**BODY, "period_end": "2026-02-28"})
    _release(client.store, "rpt-1")
    r = client.get("/reports/jobs")
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert [j["job_id"] for j in jobs] == ["rpt-2", "rpt-1"]
    assert [j["deck_ready"] for j in jobs] == [False, True]
    assert all("deck" not in j for j in jobs)
    assert client.store.recent_limit == 20


@pytest.mark.parametrize("asked, used", [("5", 5), ("0", 1), ("500", 50)])
def test_the_list_is_bounded(client, asked, used):
    client.get(f"/reports/jobs?limit={asked}")
    assert client.store.recent_limit == used


def test_the_entitlement_decision_is_filed_with_the_job(client, monkeypatch):
    """The gate's audit row carries no trace id, so the job keeps the decision it
    was filed under -- including whether shadow mode suppressed a refusal."""
    shadowed = guardrail.Decision(allowed=True, reason="denied, shadowed",
                                  policy_name="RoleDefinitionPolicy", policy_version=3,
                                  evidence={"role": "Viewer", "shadowed": True})
    monkeypatch.setattr(rr, "gate", lambda *a, **k: shadowed)
    client.post("/reports/generate", json=BODY)
    assert client.store.entitlement == {
        "action": "report.generate", "principal": CALLER, "allowed": True,
        "role": "Viewer", "policy_name": "RoleDefinitionPolicy", "policy_version": 3,
        "resolution": "resolved", "shadowed": True}


def _block(store, job_id="rpt-1"):
    store.jobs[job_id].update(status="blocked", run_id="FP-abc", stage_reached="POST_CHECK",
                              blocking=[{"finding_id": "F1", "code": "REPORT_UNTRACED_FIGURE",
                                         "severity": "HIGH", "detail": "x"}])


def test_attention_lists_blocked_and_failed_jobs_with_no_bytes(client):
    client.post("/reports/generate", json=BODY)
    client.post("/reports/generate", json={**BODY, "period_end": "2026-02-28"})
    _block(client.store)
    r = client.get("/reports/attention")
    assert r.status_code == 200
    items = r.json()["items"]
    assert [i["job_id"] for i in items] == ["rpt-1"]
    assert items[0]["status"] == "blocked" and items[0]["rerun_status"] is None
    assert items[0]["deck_ready"] is False and "deck" not in items[0]
    assert client.store.attention_limit == 50


def test_dismiss_is_gated_records_who_and_writes_the_audit_event(client, monkeypatch):
    from src.services.rga import audit
    events = []
    monkeypatch.setattr(rr.audit, "emit",
                        lambda action, **k: events.append((action, k, audit.current_context())))
    client.post("/reports/generate", json=BODY)
    _block(client.store)
    client.gates.clear()
    r = client.post("/reports/jobs/rpt-1/dismiss", json={"reason": "not needed"})
    assert r.status_code == 200, r.text
    assert r.json()["dismissed"] is True
    assert [g[0] for g in client.gates] == ["finding.resolve"]
    job = client.store.jobs["rpt-1"]
    assert (job["dismissed_by"], job["dismiss_reason"]) == (CALLER, "not needed")
    action, k, ctx = events[0]
    assert action == audit.DISMISSED
    assert k["run_id"] == "FP-abc" and k["details"]["dismissed_by"] == CALLER
    assert k["details"]["reason"] == "not needed"
    assert ctx["job_id"] == "rpt-1"


def test_dismissing_what_cannot_be_dismissed_is_409_and_writes_nothing(client, monkeypatch):
    events = []
    monkeypatch.setattr(rr.audit, "emit", lambda *a, **k: events.append(a))
    client.post("/reports/generate", json=BODY)          # queued, not blocked
    r = client.post("/reports/jobs/rpt-1/dismiss", json={})
    assert r.status_code == 409
    assert events == []
    assert client.post("/reports/jobs/rpt-nope/dismiss", json={}).status_code == 404


def test_a_refused_dismiss_changes_nothing(client, monkeypatch):
    client.post("/reports/generate", json=BODY)
    _block(client.store)

    def _refuse(*a, **k):
        raise HTTPException(status_code=403, detail="role Viewer may not perform write")

    monkeypatch.setattr(rr, "gate", _refuse)
    assert client.post("/reports/jobs/rpt-1/dismiss", json={}).status_code == 403
    assert client.store.jobs["rpt-1"].get("dismissed_at") is None


def test_attention_replies_survive_the_output_safety_boundary(client, monkeypatch):
    from src.services import output_safety as osafe
    monkeypatch.setattr(osafe, "_route_paths", {r.path for r in rr.router.routes})
    client.post("/reports/generate", json=BODY)
    _block(client.store)
    reply = client.get("/reports/attention").json()
    assert osafe.scrub_payload(reply, where="test") == reply


def test_report_signoff_is_a_transact_action():
    # transact: held by Approver and Admin, irreversible, so a stated permit is required.
    assert actions.action_class("report.signoff") == "transact"
