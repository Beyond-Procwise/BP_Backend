"""Supplier review API: list / confirm (merge) / reject (split)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import supplier_review
from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as SR

PFX = "ZZAPI"
REVIEWER = "zzapi-test-reviewer"


def _clean():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_supplier_review WHERE extracted_name ILIKE %s", (PFX + "%",))
            cur.execute("DELETE FROM proc.bp_supplier_alias WHERE alias_name ILIKE %s", (PFX + "%",))
            cur.execute("DELETE FROM proc.bp_supplier WHERE supplier_name ILIKE %s OR supplier_id ILIKE %s",
                        (PFX + "%", "SUP-" + PFX + "%"))
        c.commit()


@pytest.fixture(autouse=True)
def around():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")
    _clean(); yield; _clean()


@pytest.fixture
def client(monkeypatch):
    from api.auth import Principal, require_user

    app = FastAPI(); app.include_router(supplier_review.router)
    # confirm/reject resolve the caller since P8 phase 1, and this app never
    # configures auth, so both tests had been getting 503. They are about the
    # merge/split, not authentication, so the caller is pinned -- to a named
    # reviewer, which is the only way the screen is reached: a token without a
    # subject is refused, and an alias cannot be saved without who made it.
    app.dependency_overrides[require_user] = lambda: Principal(subject=REVIEWER)
    monkeypatch.setattr(supplier_review, "gate", lambda *a, **k: None)
    return TestClient(app)


def _recorded(review_id, alias_name):
    """(reviewed_by on the review, created_by on the alias)."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT reviewed_by FROM proc.bp_supplier_review WHERE review_id = %s",
                    (review_id,))
        reviewed_by = cur.fetchone()[0]
        cur.execute("SELECT created_by FROM proc.bp_supplier_alias "
                    "WHERE LOWER(alias_name) = LOWER(%s)", (alias_name,))
        return reviewed_by, cur.fetchone()[0]


def _mk_review(decision, chosen, candidate, name):
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_supplier_review (extracted_name, decision, chosen_supplier_id, "
                "candidate_supplier_id, candidate_supplier_name, score, status) "
                "VALUES (%s,%s,%s,%s,%s,90,'pending') RETURNING review_id",
                (name, decision, chosen, candidate, "cand"),
            )
            rid = cur.fetchone()[0]
        c.commit()
    return rid


def test_confirm_aliases_to_candidate(client):
    rid = _mk_review("created", f"SUP-{PFX}New", f"SUP-{PFX}Canon", f"{PFX} Widgets Solution Ltd")
    assert any(r["review_id"] == rid for r in client.get("/suppliers/reviews").json()["reviews"])

    res = client.post(f"/suppliers/reviews/{rid}/confirm", json={"reviewer": "t"}).json()
    assert res["supplier_id"] == f"SUP-{PFX}Canon"
    # signed by the caller's token, never by the "reviewer" the body claims
    assert _recorded(rid, f"{PFX} Widgets Solution Ltd") == (REVIEWER, REVIEWER)
    # future resolution of the variant now hits the alias → canonical
    with get_conn() as c:
        assert SR.resolve_or_create_supplier(f"{PFX} Widgets Solution Ltd", c) == f"SUP-{PFX}Canon"


def test_reject_creates_distinct_supplier(client):
    # decision=linked → it was merged into candidate; reject must give it its own id
    rid = _mk_review("linked", f"SUP-{PFX}Canon", f"SUP-{PFX}Canon", f"{PFX} Distinctco Ltd")
    res = client.post(f"/suppliers/reviews/{rid}/reject", json={"reviewer": "t"}).json()
    assert res["supplier_id"] != f"SUP-{PFX}Canon"
    assert _recorded(rid, f"{PFX} Distinctco Ltd") == (REVIEWER, REVIEWER)
    with get_conn() as c:
        assert SR.resolve_or_create_supplier(f"{PFX} Distinctco Ltd", c) == res["supplier_id"]
    # second confirm/reject on same review → 409
    assert client.post(f"/suppliers/reviews/{rid}/reject", json={"reviewer": "t"}).status_code == 409


def test_with_sign_in_off_a_decision_is_saved_and_says_nobody_was_identified(monkeypatch):
    """require_user returns None only when auth is explicitly off. That used to
    reach the alias insert as a NULL author and come back as a 500. The body's
    'reviewer' still signs nothing."""
    from api.auth import require_user

    app = FastAPI(); app.include_router(supplier_review.router)
    app.dependency_overrides[require_user] = lambda: None
    monkeypatch.setattr(supplier_review, "gate", lambda *a, **k: None)
    anonymous = TestClient(app)

    name = f"{PFX} Nobody Signed Ltd"
    rid = _mk_review("created", f"SUP-{PFX}New", f"SUP-{PFX}Canon", name)
    res = anonymous.post(f"/suppliers/reviews/{rid}/confirm", json={"reviewer": "t"})
    assert res.status_code == 200, res.text
    who = supplier_review.UNIDENTIFIED_REVIEWER
    assert _recorded(rid, name) == (who, who)

    name2 = f"{PFX} Nobody Split Ltd"
    rid2 = _mk_review("linked", f"SUP-{PFX}Canon", f"SUP-{PFX}Canon", name2)
    assert anonymous.post(f"/suppliers/reviews/{rid2}/reject", json={"reviewer": "t"}).status_code == 200
    assert _recorded(rid2, name2) == (who, who)
