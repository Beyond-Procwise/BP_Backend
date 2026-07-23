from fastapi.testclient import TestClient
from src.api.main import app
from src.services import proposal_store as ps

client = TestClient(app)

# NOTE: src/api/main.py registers its routers via the bare `api.routers` import path
# (sys.path.insert makes `api` a top-level package rooted at src/api). That means the
# module object bound into `app` is `api.routers.deal_proposals`, a DIFFERENT module
# instance from `src.api.routers.deal_proposals` even though both load the same file.
# Monkeypatching has to target whichever module the running app actually holds a
# reference to, so we patch the bare-imported one here.


def test_generate_returns_proposal_ids(monkeypatch):
    monkeypatch.setattr("api.routers.deal_proposals._generate",
                        lambda batch, session: {"batch_deal_id": batch, "proposal_ids": [101, 102],
                                                "ungrouped": [], "members_with_lines": 12, "members_total": 12})
    r = client.post("/deals/proposals/generate", json={"batch_deal_id": "BATCH1"})
    assert r.status_code == 200
    assert r.json()["proposal_ids"] == [101, 102]


def test_confirm_conflict_returns_409(monkeypatch):
    def _boom(*a, **k): raise ps.StaleProposalError("stale")
    monkeypatch.setattr("api.routers.deal_proposals._confirm", _boom)
    r = client.post("/deals/proposals/101/confirm", json={"confirmed_by": "nick"})
    assert r.status_code == 409
