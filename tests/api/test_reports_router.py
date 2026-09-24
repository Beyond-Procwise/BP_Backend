"""/reports: the Report Generation Agent's front door.

The pipeline is replaced here -- what is under test is the door, not the report:
who may open it, what they must bring, and that a blocked report never leaves.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.routers import reports as rr
from src.services import actions, guardrail
from src.services.rga.models import Finding, FindingCode, Severity
from src.services.rga.pipeline import ReportRun
from src.services.rga.render import RenderedArtefact

CALLER = "sub-real-caller"
_ALLOWED = guardrail.Decision(allowed=True, reason="compute is a reversible class",
                              policy_name="RoleDefinitionPolicy")
_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
BODY = {"report_type": "exec_procurement_summary", "period_start": "2026-01-01",
        "period_end": "2026-03-31", "period_label": "2026 Q1", "currency": "GBP"}


class _P:
    subject = CALLER


def _artefact():
    return RenderedArtefact(content=b"PK-deck-bytes", media_type=_PPTX,
                            renderer="pptx", renderer_version="1", pack_id="FP-abc",
                            pack_hash="h" * 64, style_version="s1", ast_hash="a" * 64)


def _released():
    return ReportRun(run_id="FP-abc", report_type_id="exec_procurement_summary",
                     artefact=_artefact(), released=True, stage_reached="RELEASE")


def _blocked():
    return ReportRun(
        run_id="FP-abc", report_type_id="exec_procurement_summary",
        artefact=_artefact(), released=False, stage_reached="POST_CHECK",
        findings=[
            Finding(finding_id="FP-abc-PC001", code=FindingCode.REPORT_UNTRACED_FIGURE,
                    severity=Severity.HIGH, detail="'£9,999' traces to no fact"),
            Finding(finding_id="FP-abc-F0009", code=FindingCode.REPORT_UNTRACED_FIGURE,
                    severity=Severity.LOW, detail="a note", blocks_release=False),
        ])


@pytest.fixture
def client(monkeypatch):
    gates, runs = [], []

    def _gate(action, principal, **k):
        gates.append((action, getattr(principal, "subject", None), k.get("context")))
        return _ALLOWED

    def _generate(report_type_id, **k):
        runs.append((report_type_id, k))
        return client.next_run

    monkeypatch.setattr(rr, "gate", _gate)
    monkeypatch.setattr(rr, "generate_report", _generate)
    app = FastAPI()
    app.include_router(rr.router)
    app.dependency_overrides[rr.require_user] = lambda: _P()
    client = TestClient(app)
    client.gates, client.runs, client.next_run = gates, runs, _released()
    return client


def test_report_generate_is_a_compute_action():
    # compute is reversible: building a report inside the tenant changes nothing.
    # Sending one out is report.export, a share, and stays refused by default.
    assert actions.action_class("report.generate") == "compute"
    assert actions.action_class("report.export") == "share"


def test_types_lists_what_has_a_builder(client):
    r = client.get("/reports/types")
    assert r.status_code == 200
    assert "exec_procurement_summary" in r.json()["report_types"]


def test_a_released_report_comes_back_as_the_deck(client):
    r = client.post("/reports/generate", json=BODY)
    assert r.status_code == 200, r.text
    assert r.content == b"PK-deck-bytes"
    assert r.headers["content-type"] == _PPTX
    assert r.headers["x-report-run-id"] == "FP-abc"
    assert "attachment" in r.headers["content-disposition"]
    assert "FP-abc" in r.headers["content-disposition"]

    report_type, kwargs = client.runs[0]
    assert report_type == "exec_procurement_summary"
    assert kwargs["scope"] == {"period_start": "2026-01-01", "period_end": "2026-03-31",
                               "period_label": "2026 Q1", "currency": "GBP"}


def test_the_gate_is_asked_first_and_by_the_token_holder(client):
    client.post("/reports/generate", json=BODY)
    action, subject, context = client.gates[0]
    assert (action, subject) == ("report.generate", CALLER)
    assert context["report_type"] == "exec_procurement_summary"


def test_a_refusal_stops_the_run(client, monkeypatch):
    def _refuse(*a, **k):
        raise HTTPException(status_code=403, detail="refused by a rule")

    monkeypatch.setattr(rr, "gate", _refuse)
    r = client.post("/reports/generate", json=BODY)
    assert r.status_code == 403
    assert client.runs == []


def test_a_blocked_report_returns_reasons_and_never_the_deck(client):
    client.next_run = _blocked()
    r = client.post("/reports/generate", json=BODY)
    assert r.status_code == 422
    assert b"PK-deck-bytes" not in r.content
    body = r.json()
    assert body["released"] is False
    assert body["run_id"] == "FP-abc"
    assert body["stage_reached"] == "POST_CHECK"
    # Only what blocked it: a non-blocking note is not a reason.
    assert [f["finding_id"] for f in body["blocking"]] == ["FP-abc-PC001"]
    assert body["blocking"][0]["code"] == "REPORT_UNTRACED_FIGURE"


def test_an_unknown_report_type_is_refused_before_anything_runs(client):
    r = client.post("/reports/generate", json={**BODY, "report_type": "board_paper"})
    assert r.status_code == 404
    assert client.runs == [] and client.gates == []


@pytest.mark.parametrize("patch", [
    {"period_start": "2026-04-01"},             # starts after it ends
    {"period_end": "not-a-date"},
    {"currency": "pounds"},
])
def test_a_malformed_period_is_refused(client, patch):
    r = client.post("/reports/generate", json={**BODY, **patch})
    assert r.status_code == 422
    assert client.runs == []


def test_label_and_currency_have_defaults(client):
    body = {k: v for k, v in BODY.items() if k not in ("period_label", "currency")}
    assert client.post("/reports/generate", json=body).status_code == 200
    scope = client.runs[0][1]["scope"]
    assert scope["currency"] == "GBP"
    assert scope["period_label"] == "2026-01-01 to 2026-03-31"


def test_an_anonymous_caller_is_refused_when_auth_is_on(monkeypatch):
    # No override of require_user: the real dependency runs.
    monkeypatch.setattr(rr, "generate_report", lambda *a, **k: _released())
    import api.auth as auth

    class _V:
        def verify(self, token):
            raise auth.AuthError("bad")

    monkeypatch.setattr(auth, "_active_verifier", lambda: _V())
    app = FastAPI()
    app.include_router(rr.router)
    r = TestClient(app).post("/reports/generate", json=BODY)
    assert r.status_code == 401
