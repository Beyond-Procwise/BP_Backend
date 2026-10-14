"""Supplier web-research: grounding guard, fill-empty-only rules, sensitive-field guard.

The propose-only behaviour itself lives in test_supplier_research_propose_only.py.
"""
import pytest

from src.services.db import get_conn
from src.services.supplier_enrichment import research as R

IDP = "SUP-ZZRES"


def _clean():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_supplier_enrichment WHERE supplier_id ILIKE %s", (IDP+"%",))
            cur.execute("DELETE FROM proc.bp_supplier WHERE supplier_id ILIKE %s", (IDP+"%",))
        c.commit()


@pytest.fixture(autouse=True)
def around():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")
    _clean(); yield; _clean()


# ---- pure functions (no DB) ----
def test_parse_result_extracts_matched_name_and_fields():
    c = ('ok {"matched_name":"X Ltd","fields": {"website_url": '
         '{"value":"https://x.com","source_url":"https://x.com","confidence":0.9}}}}')  # note extra brace
    matched, fields = R._parse_result(c)
    assert matched == "X Ltd"
    assert fields["website_url"]["value"] == "https://x.com"


def test_ground_drops_uncited_sensitive_and_unknown():
    # _ground now takes url -> the text the model was shown for that url, not a bare URL set:
    # a page that was visited but yielded nothing must not ground anything. See
    # test_supplier_research_grounding.py for the content-verification behaviour itself.
    evidence = {
        "https://acme.com/about": "Acme Ltd is based in London. Visit acme.com for more.",
        "https://acme.com": "Acme Ltd home page.",
    }
    fields = {
        "website_url": {"value": "https://acme.com", "source_url": "https://acme.com/about", "confidence": 0.9},
        "city": {"value": "London", "source_url": "https://evil.example/x", "confidence": 0.9},  # page never seen
        "vat_number": {"value": "GB123", "source_url": "https://acme.com", "confidence": 0.9},   # sensitive
        "country": {"value": "unknown", "source_url": "https://acme.com", "confidence": 0.9},     # unknown
    }
    kept = R._ground(fields, evidence)
    assert set(kept) == {"website_url"}


# ---- DB-backed apply / end-to-end (mocked research loop) ----
def _seed(sid, name, **cols):
    keys = ["supplier_id", "supplier_name", "trading_name"] + list(cols)
    vals = [sid, name, name] + list(cols.values())
    ph = ",".join(["%s"] * len(keys))
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(f"INSERT INTO proc.bp_supplier ({','.join(keys)}, created_date, created_by) VALUES ({ph}, NOW(), 'test')", vals)
        c.commit()


def test_only_empty_non_sensitive_fields_are_proposed(monkeypatch):
    """Was test_apply_fills_empty_only_when_entity_matches, and asserted that a
    confident, well-matched result was written straight into bp_supplier.

    Research proposes now (A19.39) — see test_supplier_research_propose_only.py.
    What this still pins is which fields make it into the proposal at all: the
    empty one yes, the one that already has a value no, the sensitive one never.
    """
    _seed(f"{IDP}1", "Acme Widgets Ltd", country="GB")  # country set, website empty
    canned = ('{"matched_name":"Acme Widgets Ltd","fields": {'
              '"website_url": {"value":"https://acme.example","source_url":"https://acme.example/about","confidence":0.9},'
              '"country": {"value":"US","source_url":"https://acme.example/about","confidence":0.9},'
              '"vat_number": {"value":"GB999","source_url":"https://acme.example/about","confidence":0.9}}}')
    # _run_loop returns url -> the text the model was shown. The page has to actually
    # contain the claimed values now, which is the point of the content check.
    monkeypatch.setattr(R, "_run_loop", lambda name: (canned, {
        "https://acme.example/about":
            "Acme Widgets Ltd, US. Official site: acme.example",
    }))

    with get_conn() as c:
        res = R.research_and_enrich(f"{IDP}1", c)
    assert res["entity_confirmed"] is True
    assert "website_url" in res["proposed"]          # empty → offered
    assert "country" not in res["proposed"]          # non-empty → never overwritten
    assert "vat_number" not in res["fields"]         # sensitive → never even kept
    assert res["applied"] == {}                      # and nothing was written
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT website_url, country FROM proc.bp_supplier WHERE supplier_id=%s", (f"{IDP}1",))
        website, country = cur.fetchone()
    assert website in (None, "") and country == "GB"


def test_entity_mismatch_stays_pending(monkeypatch):
    # Found a real, well-cited company — but a DIFFERENT one. Nothing is written
    # either way now; what this pins is that the mismatch is still detected and
    # reported, rather than quietly dropped along with the gate it used to feed.
    _seed(f"{IDP}3", "Zephyr Robotics Ltd")  # all empty
    canned = ('{"matched_name":"Umbrella Facilities Management Ltd","fields": {'
              '"website_url": {"value":"https://umbrella.example","source_url":"https://umbrella.example","confidence":0.95},'
              '"country": {"value":"United Kingdom","source_url":"https://umbrella.example","confidence":0.95}}}')
    monkeypatch.setattr(R, "_run_loop", lambda name: (canned, {
        "https://umbrella.example":
            "Umbrella Facilities Management Ltd, United Kingdom. umbrella.example",
    }))
    with get_conn() as c:
        res = R.research_and_enrich(f"{IDP}3", c)
    assert res["entity_confirmed"] is False and res["applied"] == {}  # cited but wrong entity
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT website_url FROM proc.bp_supplier WHERE supplier_id=%s", (f"{IDP}3",))
        assert cur.fetchone()[0] in (None, "")  # nothing written


def test_uncited_fields_not_applied(monkeypatch):
    _seed(f"{IDP}2", "Beta Traders Ltd")  # all empty
    canned = '{"matched_name":"Beta Traders Ltd","fields": {"website_url": {"value":"https://beta.example","source_url":"https://hallucinated.example","confidence":0.95}}}'
    monkeypatch.setattr(R, "_run_loop", lambda name: (canned, {
        "https://realsource.example": "Beta Traders Ltd. Official site: beta.example",
    }))  # cited page ≠ the page actually read
    with get_conn() as c:
        res = R.research_and_enrich(f"{IDP}2", c)
    assert res["fields"] == {} and res["applied"] == {}  # dropped as ungrounded
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT website_url FROM proc.bp_supplier WHERE supplier_id=%s", (f"{IDP}2",))
        assert cur.fetchone()[0] in (None, "")


# ---- human approve + review-list payload ----
def _seed_enrichment(supplier_id, fields, matched_name, name_match, status="pending"):
    import json as _j
    from src.services.db import get_conn as _gc
    with _gc() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_supplier_enrichment (supplier_id, model, fields, citations, confidence, raw, apply_status, applied_fields) "
                "VALUES (%s,'test',%s::jsonb,%s::jsonb,%s,%s::jsonb,%s,'{}'::jsonb) RETURNING enrichment_id",
                (supplier_id, _j.dumps(fields), _j.dumps(["https://acme.example/about"]), 0.9,
                 _j.dumps({"matched_name": matched_name, "name_match": name_match}), status),
            )
            eid = cur.fetchone()[0]
        c.commit()
    return eid


def test_apply_enrichment_fills_empty(monkeypatch):
    _seed(f"{IDP}4", "Acme Widgets Ltd")  # all empty
    fields = {"website_url": {"value": "https://acme.example", "source_url": "https://acme.example/about", "confidence": 0.9}}
    eid = _seed_enrichment(f"{IDP}4", fields, "Acme Widgets Ltd", 100.0)
    with get_conn() as c:
        res = R.apply_enrichment(eid, "reviewer1", c)
    assert res["status"] == "applied" and "website_url" in res["applied"]
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT website_url FROM proc.bp_supplier WHERE supplier_id=%s", (f"{IDP}4",))
        assert cur.fetchone()[0] == "https://acme.example"


def test_enrichment_review_payload_has_matched_name_and_citations():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routers import supplier_research as SRR
    _seed(f"{IDP}5", "Beta Corp Ltd")
    fields = {"website_url": {"value": "https://beta.example", "source_url": "https://acme.example/about", "confidence": 0.9}}
    eid = _seed_enrichment(f"{IDP}5", fields, "Beta Corp Limited", 96.0)

    app = FastAPI(); app.include_router(SRR.router)
    data = TestClient(app).get("/suppliers/enrichment/reviews?status=pending").json()
    mine = [r for r in data["reviews"] if r["enrichment_id"] == eid]
    assert mine, "pending enrichment should appear in the review payload"
    r = mine[0]
    assert r["matched_name"] == "Beta Corp Limited"
    assert r["citations"] and "would_fill" in r and "website_url" in r["would_fill"]
    assert "current" in r


def test_reviews_queue_combines_match_and_enrichment():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routers import supplier_review as SRV
    # a supplier-match review
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_supplier_review (extracted_name, decision, chosen_supplier_id, "
                "candidate_supplier_id, candidate_supplier_name, score, status) "
                "VALUES (%s,'existing_dup',%s,%s,%s,90,'pending') RETURNING review_id",
                (f"{IDP}Q A", f"SUP-{IDP}QA", f"SUP-{IDP}QB", f"{IDP}Q B"))
            mrid = cur.fetchone()[0]
        c.commit()
    # an enrichment approval
    _seed(f"{IDP}Q1", "Queue Corp Ltd")
    eid = _seed_enrichment(f"{IDP}Q1", {"website_url": {"value": "https://q.example", "source_url": "https://acme.example/about", "confidence": 0.9}}, "Queue Corp Limited", 96.0)
    try:
        app = FastAPI(); app.include_router(SRV.router)
        q = TestClient(app).get("/suppliers/reviews/queue").json()
        by_id = {(i["review_type"], i["id"]): i for i in q["items"]}
        assert ("supplier_match", mrid) in by_id
        enr = by_id[("supplier_enrichment", eid)]
        assert "website_url" in enr["detail"]["would_fill"]
        assert any("/apply" in a["path"] for a in enr["actions"])
        assert any("/confirm" in a["path"] for a in by_id[("supplier_match", mrid)]["actions"])
    finally:
        with get_conn() as c:
            with c.cursor() as cur:
                cur.execute("DELETE FROM proc.bp_supplier_review WHERE review_id=%s", (mrid,))
            c.commit()
