from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import summary as summary_router
import src.services.summary_agent as sa


def _client():
    from api.auth import require_user

    app = FastAPI()
    app.include_router(summary_router.router)
    # The POST routes resolve the caller (P8 phase 2); these tests are about
    # the summaries, not authentication, so the principal is pinned: nobody.
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app)


def test_post_summary_generates(monkeypatch):
    monkeypatch.setattr(
        sa, "generate_summary",
        lambda persona, deal_id=None, as_of=None: {
            "summary_id": "sid-1", "persona": persona, "scope": "portfolio",
            "deal_id": None, "summary": "S", "sources": {}, "generated_at": "t",
        },
    )
    resp = _client().post("/summary", json={"persona": "analysis"})
    assert resp.status_code == 200
    assert resp.json()["summary_id"] == "sid-1"


def test_post_summary_404_when_no_data(monkeypatch):
    monkeypatch.setattr(sa, "generate_summary", lambda persona, deal_id=None, as_of=None: None)
    resp = _client().post("/summary", json={"persona": "analysis", "deal_id": "NOPE"})
    assert resp.status_code == 404


def test_get_summary_returns_cache(monkeypatch):
    monkeypatch.setattr(
        sa, "get_cached_summary",
        lambda persona, deal_id=None: {"summary_id": "sid-1", "summary": "cached"},
    )
    resp = _client().get("/summary", params={"persona": "analysis"})
    assert resp.status_code == 200
    assert resp.json()["summary"] == "cached"


def test_get_summary_404_when_uncached(monkeypatch):
    monkeypatch.setattr(sa, "get_cached_summary", lambda persona, deal_id=None: None)
    resp = _client().get("/summary", params={"persona": "analysis"})
    assert resp.status_code == 404


def test_get_history(monkeypatch):
    monkeypatch.setattr(sa, "list_summary_history", lambda persona, deal_id=None: [{"summary_id": "a"}])
    resp = _client().get("/summary/history", params={"persona": "analysis"})
    assert resp.status_code == 200
    assert resp.json()["history"] == [{"summary_id": "a"}]


def test_post_precompute(monkeypatch):
    monkeypatch.setattr(sa, "precompute_summaries", lambda personas=None, deal_ids=None: {"generated": 4, "failed": 0})
    resp = _client().post("/summary/precompute", json={})
    assert resp.status_code == 200
    assert resp.json()["generated"] == 4
