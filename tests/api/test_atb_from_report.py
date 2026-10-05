"""POST /atb/import/from-report/{job_id}: the style of a released report, as a candidate pack."""
import pytest
from fastapi import HTTPException

from api.endpoint_gate import NotPermitted
from api.routers import atb as ar
from src.services.atb.pptx_import.import_pack import ImportRefused, ImportResult
from tests.api.test_atb_router import PACK, client  # noqa: F401

JOB_ID = "rpt-abcdef123456"
JOB = {"job_id": JOB_ID, "report_type": "exec_procurement_summary"}
URL = f"/atb/import/from-report/{JOB_ID}"


def _result(pack_id="p-9", version=1):
    return ImportResult(pack_key=f"from-report-{JOB_ID}", version=version, pack=PACK,
                        layouts=[], single_use=[], evidence={}, pack_id=pack_id)


@pytest.fixture
def wired(client, monkeypatch):
    seen = {"import": [], "notes": []}
    monkeypatch.setattr(ar, "readable_deck",
                        lambda job_id, principal: (JOB, b"PK-deck", "application/x", "d.pptx"))

    def _import(data, filename, user, conn=None):
        seen["import"].append((data, filename, user))
        return _result()
    monkeypatch.setattr(ar, "import_pack", _import)
    monkeypatch.setattr(ar.store, "set_pack_notes",
                        lambda conn, pack_id, notes: seen["notes"].append((pack_id, notes)),
                        raising=False)
    client.seen = seen
    return client


def test_it_is_gated_as_a_write(wired):
    assert wired.post(URL).status_code == 200
    assert wired.gates == ["style_pack.write"]


def test_the_stored_deck_is_what_the_importer_reads_under_a_per_report_name(wired):
    wired.post(URL)
    data, filename, user = wired.seen["import"][0]
    assert data == b"PK-deck"
    assert filename == f"from-report-{JOB_ID}.pptx"
    assert user == "sub-1"


def test_the_pack_says_where_it_came_from(wired):
    wired.post(URL)
    pack_id, notes = wired.seen["notes"][0]
    assert pack_id == "p-9"
    assert "generated report" in notes and JOB_ID in notes
    assert "not necessarily how its source style pack was defined" in notes


def test_a_caller_who_cannot_read_the_deck_cannot_learn_its_style(wired, monkeypatch):
    def refuse(job_id, principal):
        raise HTTPException(status_code=409,
                            detail=f"report job {job_id} is awaiting sign-off; only a person "
                                   "who may sign it off can open it before then")
    monkeypatch.setattr(ar, "readable_deck", refuse)
    r = wired.post(URL)
    assert r.status_code == 409 and "awaiting sign-off" in r.json()["detail"]
    assert wired.seen["import"] == [] and wired.seen["notes"] == []


def test_a_caller_without_style_pack_write_is_refused_before_the_deck_is_touched(wired, monkeypatch):
    def deny(action, *a, **k):
        raise NotPermitted(action)
    touched = []
    monkeypatch.setattr(ar, "gate", deny)
    monkeypatch.setattr(ar, "readable_deck", lambda *a: touched.append(1))
    assert wired.post(URL).status_code == 403
    assert touched == []


def test_an_unreadable_deck_is_a_400_in_the_importers_words_and_stores_nothing(wired, monkeypatch):
    def refuse(*a, **k):
        raise ImportRefused("the presentation has no slides")
    monkeypatch.setattr(ar, "import_pack", refuse)
    r = wired.post(URL)
    assert r.status_code == 400 and r.json()["detail"] == "the presentation has no slides"
    assert wired.seen["notes"] == []


def test_the_same_report_twice_is_a_new_version_of_one_pack(wired, monkeypatch):
    results = iter([_result("p-9", 1), _result("p-10", 2)])
    monkeypatch.setattr(ar, "import_pack", lambda *a, **k: next(results))
    a = wired.post(URL).json()
    b = wired.post(URL).json()
    assert a["pack_key"] == b["pack_key"] and (a["version"], b["version"]) == (1, 2)


def test_the_pack_is_never_approved_by_this_route(wired, monkeypatch):
    approved = []
    monkeypatch.setattr(ar.store, "set_pack_status", lambda *a, **k: approved.append(a))
    monkeypatch.setattr(ar.store, "set_layout_status", lambda *a, **k: approved.append(a))
    wired.post(URL)
    assert approved == []


def test_a_pack_summary_exposes_its_notes(client):
    pack = client.get("/atb/packs").json()["packs"][0]
    assert "notes" in pack
