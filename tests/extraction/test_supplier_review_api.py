"""Supplier review API: list / confirm (merge) / reject (split)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import supplier_review
from src.services.db import get_conn
from src.services.extraction_v3 import supplier_resolver as SR

PFX = "ZZAPI"


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
def client():
    from api.auth import require_user

    app = FastAPI(); app.include_router(supplier_review.router)
    # confirm/reject resolve the caller since P8 phase 1, and this app never
    # configures auth, so both tests had been getting 503. They are about the
    # merge/split, not authentication: the principal is pinned, nobody.
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app)


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
    # future resolution of the variant now hits the alias → canonical
    with get_conn() as c:
        assert SR.resolve_or_create_supplier(f"{PFX} Widgets Solution Ltd", c) == f"SUP-{PFX}Canon"


def test_reject_creates_distinct_supplier(client):
    # decision=linked → it was merged into candidate; reject must give it its own id
    rid = _mk_review("linked", f"SUP-{PFX}Canon", f"SUP-{PFX}Canon", f"{PFX} Distinctco Ltd")
    res = client.post(f"/suppliers/reviews/{rid}/reject", json={"reviewer": "t"}).json()
    assert res["supplier_id"] != f"SUP-{PFX}Canon"
    with get_conn() as c:
        assert SR.resolve_or_create_supplier(f"{PFX} Distinctco Ltd", c) == res["supplier_id"]
    # second confirm/reject on same review → 409
    assert client.post(f"/suppliers/reviews/{rid}/reject", json={"reviewer": "t"}).status_code == 409
