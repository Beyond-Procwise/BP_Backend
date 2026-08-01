"""Analysis-event routes."""
import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

mod = importlib.import_module("src.api.routers.analysis")


def _client():
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app)


def test_start_returns_the_analysis_id(monkeypatch):
    monkeypatch.setattr(mod.analysis_store, "start", lambda **kw: "aid-1")
    r = _client().post("/analysis", json={"session_id": "ses-1",
                                          "name": "Q3", "mode": "new"})
    assert r.status_code == 200
    assert r.json() == {"analysis_id": "aid-1"}


def test_start_rejects_a_missing_session_id():
    r = _client().post("/analysis", json={"name": "Q3"})
    assert r.status_code == 422


def test_by_deal_and_by_session_are_declared_before_the_id_route():
    """Otherwise /{analysis_id} swallows them."""
    paths = [r.path for r in mod.router.routes]
    assert paths.index("/analysis/by-deal/{deal_id}") < paths.index("/analysis/{analysis_id}")
    assert paths.index("/analysis/by-session/{session_id}") < paths.index("/analysis/{analysis_id}")


def test_by_deal_returns_versions_newest_first_with_deltas(monkeypatch):
    monkeypatch.setattr(mod, "_query", lambda sql, params: [
        {"analysis_id": "a2", "version": 2, "name": "v2",
         "started_at": "2026-07-14", "document_count": 2, "value_found": 9100,
         "currency": "GBP", "status": "complete"},
        {"analysis_id": "a1", "version": 1, "name": "v1",
         "started_at": "2026-07-02", "document_count": 6, "value_found": 9100,
         "currency": "GBP", "status": "complete"},
    ])
    body = _client().get("/analysis/by-deal/D-1").json()

    assert [v["version"] for v in body["versions"]] == [2, 1]
    assert body["versions"][0]["delta"] == {"document_count": -4,
                                            "value_found": 0.0}
    assert body["versions"][1]["delta"] is None   # nothing to compare v1 to


def test_a_deal_with_no_analyses_returns_an_empty_list_not_an_error(monkeypatch):
    """5,040 of 5,043 deals are in this state — it is the common path."""
    monkeypatch.setattr(mod, "_query", lambda sql, params: [])
    body = _client().get("/analysis/by-deal/D-1").json()
    assert body == {"deal_id": "D-1", "versions": []}


def test_get_by_session_404s_when_there_is_no_such_analysis(monkeypatch):
    monkeypatch.setattr(mod, "_query", lambda sql, params: [])
    assert _client().get("/analysis/by-session/nope").status_code == 404
