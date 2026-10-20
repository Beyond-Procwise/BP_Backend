"""/reports/generate for the supplier QBR: written for one supplier over one period."""
import pytest

from api.routers import reports as rr
from tests.api.test_reports_router import BODY, client  # noqa: F401  (the fixture and a valid body)

QBR = {"report_type": "supplier_qbr", "supplier_id": "S1", "period_start": "2026-07-01",
       "period_end": "2026-09-30", "period_label": "Q3 2026", "currency": "GBP"}


@pytest.fixture(autouse=True)
def _supplier_known(monkeypatch):
    monkeypatch.setattr(rr, "_supplier_exists", lambda sid: sid == "S1", raising=False)


def test_the_type_is_offered(client):
    assert "supplier_qbr" in client.get("/reports/types").json()["report_types"]


def test_a_supplier_review_is_filed_with_the_supplier_in_its_scope(client):
    r = client.post("/reports/generate", json=QBR)
    assert r.status_code == 202, r.text
    report_type, scope, _as_of, _by = client.store.created[0]
    assert report_type == "supplier_qbr"
    assert scope == {"supplier_id": "S1", "period_start": "2026-07-01", "period_end": "2026-09-30",
                     "period_label": "Q3 2026", "currency": "GBP"}


def test_the_supplier_is_trimmed(client):
    client.post("/reports/generate", json={**QBR, "supplier_id": "  S1  "})
    assert client.store.created[0][1]["supplier_id"] == "S1"


@pytest.mark.parametrize("patch", [{"supplier_id": None}, {"supplier_id": "   "}, {"supplier_id": ""}])
def test_a_supplier_is_required(client, patch):
    r = client.post("/reports/generate", json={**QBR, **patch})
    assert r.status_code == 422
    assert client.store.created == [] and client.gates == []


@pytest.mark.parametrize("patch", [{"period_start": "2026-10-01"}, {"period_end": "nope"},
                                   {"period_start": None}])
def test_the_period_is_still_required_and_ordered(client, patch):
    r = client.post("/reports/generate", json={**QBR, **patch})
    assert r.status_code == 422
    assert client.store.created == []


def test_an_unknown_supplier_is_404_before_the_gate_or_a_job(client):
    r = client.post("/reports/generate", json={**QBR, "supplier_id": "NOPE"})
    assert r.status_code == 404
    assert client.store.created == [] and client.gates == []


def test_the_gate_sees_the_supplier_in_the_scope(client):
    client.post("/reports/generate", json=QBR)
    _action, _subject, context = client.gates[0]
    assert context["scope"]["supplier_id"] == "S1"


def test_other_report_types_do_not_take_a_supplier(client):
    r = client.post("/reports/generate", json={**BODY, "supplier_id": "S1"})
    assert r.status_code == 202
    assert "supplier_id" not in client.store.created[0][1]
