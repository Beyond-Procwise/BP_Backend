from fastapi.testclient import TestClient
from src.api.main import app
from src.services import proposal_store as ps
from src.api.routers.deal_proposals import _drop_rejected_pairings

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


# --- FIX I2: a rejected pairing must never be re-proposed --------------------------

class _RejectedPairCur:
    """Fake cursor for proposal_store.rejected_member_sets: one rejected proposal
    covering the quote pair {Q1, Q2}."""
    def __init__(self, rows):
        self._rows = rows
    def execute(self, sql, params=()):
        pass
    def fetchall(self):
        return self._rows
    description = [("proposal_id",), ("doc_pk",)]


def _quote_members(*pks):
    return [{"doc_type": "quote", "doc_pk": pk, "base_reference": pk,
             "role": "anchor_quote", "match_score": None, "match_evidence": None}
            for pk in pks]


def test_drop_rejected_pairings_drops_exact_rejected_pair_keeps_others():
    cur = _RejectedPairCur([(9, "Q1"), (9, "Q2")])   # rejected proposal covered {Q1, Q2}
    proposals = [
        {"proposed_name": "Re-formed same pair", "members": _quote_members("Q1", "Q2")},
        {"proposed_name": "Different pairing", "members": _quote_members("Q1", "Q3")},
    ]
    kept = _drop_rejected_pairings(cur, "BATCH1", proposals)
    assert len(kept) == 1
    assert kept[0]["proposed_name"] == "Different pairing"


def test_drop_rejected_pairings_noop_when_no_rejections():
    cur = _RejectedPairCur([])
    proposals = [{"proposed_name": "Anything", "members": _quote_members("Q1", "Q2")}]
    kept = _drop_rejected_pairings(cur, "BATCH1", proposals)
    assert kept == proposals
