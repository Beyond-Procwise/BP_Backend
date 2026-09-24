"""The report editor through the WHOLE app -- its exception handler and output-safety
middleware included. The router's own tests use a bare app, which is how two faults shipped
(final review): the scrubber rewrote a paragraph mentioning a 'pipeline' into its apology
sentence, and the 422's reasons were flattened into a Python repr the screen could not read."""
import pytest
from fastapi.testclient import TestClient

from api.routers import reports as rr
from src.services import guardrail

PROSE = "Container freight spend came to {{F0001}} across the sourcing pipeline."


class _P:
    subject = "sub-editor"


class Editing:
    from src.services.rga.editing import EditRefused, NotEditable, StaleVersion

    def draft(self, job_id):
        return {"version": 1, "title": "Spend/supplier/detail review", "facts": [],
                "editable": True, "reason": None,
                "ast": {"sections": [{"id": "s", "title": "Spend/supplier/detail",
                                      "blocks": [{"type": "narrative", "text": PROSE,
                                                  "fact_refs": ["F0001"]}]}]}}

    def save(self, job_id, **kw):
        raise self.EditRefused(['In "Pipeline review", block 2: Sentences can\'t contain '
                                "typed numbers — insert a figure instead."])


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setattr(rr, "editing", Editing())
    monkeypatch.setattr(rr, "gate", lambda *a, **k: guardrail.Decision(
        allowed=True, reason="t", policy_name="t"))
    app.dependency_overrides[rr.require_user] = lambda: _P()
    yield TestClient(app)
    app.dependency_overrides.pop(rr.require_user, None)


def test_the_draft_comes_back_word_for_word(client):
    body = client.get("/reports/jobs/rpt-1/draft").json()
    assert body["ast"]["sections"][0]["blocks"][0]["text"] == PROSE
    assert body["ast"]["sections"][0]["title"] == "Spend/supplier/detail"
    assert body["title"] == "Spend/supplier/detail review"


def test_a_refused_save_arrives_as_a_list_of_reasons(client):
    r = client.post("/reports/jobs/rpt-1/versions",
                    json={"base_version": 1, "title": "T", "ast": {"sections": []}})
    assert r.status_code == 422
    assert r.json()["detail"]["reasons"] == [
        'In "Pipeline review", block 2: Sentences can\'t contain typed numbers — '
        "insert a figure instead."]


def test_other_report_answers_are_still_scrubbed(client, monkeypatch):
    """The exemption is the editor's document only: a job's own answer is still scrubbed."""
    class Store:
        def get(self, job_id):
            return {"job_id": job_id, "status": "failed", "error": "psycopg2 error on proc.bp_report_job"}
    monkeypatch.setattr(rr, "job_store", Store())
    monkeypatch.setattr(rr, "signoff", type("S", (), {"state": staticmethod(
        lambda job: {"state": "not_released"})})())
    assert "bp_report_job" not in client.get("/reports/jobs/rpt-1").text
