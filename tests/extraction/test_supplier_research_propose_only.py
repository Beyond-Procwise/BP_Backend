"""Supplier research proposes; a human applies. It does not write on its own.

`research_and_enrich` auto-applied researched facts straight into
`proc.bp_supplier` whenever the model's self-reported confidence cleared
SUPPLIER_RESEARCH_APPLY_CONF (0.75, environment-tunable) and a fuzzy name match
cleared 85. Two numbers, one of them the model marking its own homework, and a
web page could change the supplier master.

The agreed decision (A19.39) is propose-only. Everything researched lands in the
enrichment review queue, which already has apply and reject endpoints, and the
supplier master is written only by the human path.

This is a REMOVAL. The tests below are mostly about what no longer happens, so
each one pins the observable consequence -- the row in `proc.bp_supplier` -- not
the absence of a function call.
"""
import json

import pytest

from src.services.db import get_conn
from src.services.supplier_enrichment import research as R

IDP = "SUP-ZZPROP"


def _clean():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_supplier_enrichment WHERE supplier_id ILIKE %s", (IDP + "%",))
            cur.execute("DELETE FROM proc.bp_supplier WHERE supplier_id ILIKE %s", (IDP + "%",))
        c.commit()


@pytest.fixture(autouse=True)
def around():
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")
    _clean(); yield; _clean()


def _seed(sid, name, **cols):
    keys = ["supplier_id", "supplier_name", "trading_name"] + list(cols)
    vals = [sid, name, name] + list(cols.values())
    ph = ",".join(["%s"] * len(keys))
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                f"INSERT INTO proc.bp_supplier ({','.join(keys)}, created_date, created_by) "
                f"VALUES ({ph}, NOW(), 'test')", vals)
        c.commit()


def _supplier(sid, cols):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(f"SELECT {', '.join(cols)} FROM proc.bp_supplier WHERE supplier_id = %s", (sid,))
        return dict(zip(cols, cur.fetchone()))


def _enrichment(sid):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT enrichment_id, apply_status, applied_fields, fields "
            "FROM proc.bp_supplier_enrichment WHERE supplier_id = %s "
            "ORDER BY created_date DESC LIMIT 1", (sid,))
        row = cur.fetchone()
    if not row:
        return None
    eid, status, applied, fields = row
    if isinstance(applied, str):
        applied = json.loads(applied or "{}")
    if isinstance(fields, str):
        fields = json.loads(fields or "{}")
    return {"enrichment_id": eid, "apply_status": status,
            "applied_fields": applied, "fields": fields}


# A well-cited, high-confidence, correctly-matched result: everything the old
# auto-apply gate asked for. Under propose-only it still must not write.
_CANNED = ('{"matched_name":"Acme Widgets Ltd","fields": {'
           '"website_url": {"value":"https://acme.example","source_url":"https://acme.example/about","confidence":0.95},'
           '"city": {"value":"London","source_url":"https://acme.example/about","confidence":0.95}}}')
_EVIDENCE = {"https://acme.example/about":
             "Acme Widgets Ltd of London. Official site: acme.example"}


# ---------------------------------------------------------------------------
# the removal
# ---------------------------------------------------------------------------
def test_a_high_confidence_fact_is_proposed_and_not_written(monkeypatch):
    """The finding itself: this currently fills bp_supplier.website_url."""
    sid = f"{IDP}1"
    _seed(sid, "Acme Widgets Ltd")
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))

    with get_conn() as c:
        res = R.research_and_enrich(sid, c)

    row = _supplier(sid, ["website_url", "city"])
    assert row["website_url"] in (None, ""), (
        f"research wrote to the supplier master on its own: {row}")
    assert row["city"] in (None, ""), f"research wrote to the supplier master: {row}"
    assert res["proposed"], "a grounded, confident fact should be proposed for review"
    assert "website_url" in res["proposed"]


def test_the_enrichment_is_left_pending_for_a_human(monkeypatch):
    sid = f"{IDP}2"
    _seed(sid, "Acme Widgets Ltd")
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))

    with get_conn() as c:
        R.research_and_enrich(sid, c)

    stored = _enrichment(sid)
    assert stored["apply_status"] == "pending", (
        f"an unreviewed enrichment is not 'applied': {stored['apply_status']}")
    assert stored["applied_fields"] in ({}, None), stored["applied_fields"]


def test_research_reports_nothing_as_applied(monkeypatch):
    """The return value is what the API hands back. It must not claim a write
    that did not happen."""
    sid = f"{IDP}3"
    _seed(sid, "Acme Widgets Ltd")
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))

    with get_conn() as c:
        res = R.research_and_enrich(sid, c)

    assert res["applied"] == {}


def test_no_confidence_however_high_reopens_the_write_path(monkeypatch):
    """Certainty was the old gate. It is not a gate any more, because the number
    is the model's opinion of itself."""
    sid = f"{IDP}4"
    _seed(sid, "Acme Widgets Ltd")
    certain = _CANNED.replace("0.95", "1.0")
    monkeypatch.setattr(R, "_run_loop", lambda name: (certain, _EVIDENCE))

    with get_conn() as c:
        R.research_and_enrich(sid, c)

    assert _supplier(sid, ["website_url"])["website_url"] in (None, "")


def test_an_entity_mismatch_is_still_recorded_for_the_reviewer(monkeypatch):
    """Propose-only does not mean 'stop checking'. The name match is evidence a
    reviewer needs; it just no longer decides anything on its own."""
    sid = f"{IDP}5"
    _seed(sid, "Zephyr Robotics Ltd")
    other = ('{"matched_name":"Umbrella Facilities Management Ltd","fields": {'
             '"website_url": {"value":"https://umbrella.example","source_url":"https://umbrella.example","confidence":0.95}}}')
    monkeypatch.setattr(R, "_run_loop", lambda name: (other, {
        "https://umbrella.example": "Umbrella Facilities Management Ltd. umbrella.example"}))

    with get_conn() as c:
        res = R.research_and_enrich(sid, c)

    assert res["entity_confirmed"] is False
    assert res["name_match"] < R._ENTITY_MATCH_FLOOR
    assert _supplier(sid, ["website_url"])["website_url"] in (None, "")


# ---------------------------------------------------------------------------
# _SENSITIVE must not regress -- it is already correct and stays correct
# ---------------------------------------------------------------------------
def test_sensitive_fields_are_never_researched():
    """Not "researched then filtered": they are not in the field list the model
    is asked for, and the system prompt tells it not to report them."""
    assert not (set(R._RESEARCH_FIELDS) & R._SENSITIVE)
    assert not (set(R._APPLY_COLUMNS) & R._SENSITIVE)
    for banned in ("bank", "tax", "VAT", "registration"):
        assert banned.lower() in R._SYSTEM.lower()


def test_a_sensitive_fact_is_dropped_even_when_perfectly_grounded(monkeypatch):
    """The model reports one anyway, cited and confident. It must not survive
    into the record, the proposal, or the review queue."""
    sid = f"{IDP}6"
    _seed(sid, "Acme Widgets Ltd")
    leaky = ('{"matched_name":"Acme Widgets Ltd","fields": {'
             '"website_url": {"value":"https://acme.example","source_url":"https://acme.example/about","confidence":0.95},'
             '"bank_iban": {"value":"GB29NWBK60161331926819","source_url":"https://acme.example/about","confidence":0.99},'
             '"vat_number": {"value":"GB123456789","source_url":"https://acme.example/about","confidence":0.99}}}')
    monkeypatch.setattr(R, "_run_loop", lambda name: (leaky, {
        "https://acme.example/about":
            "Acme Widgets Ltd. IBAN GB29NWBK60161331926819. VAT GB123456789. acme.example"}))

    with get_conn() as c:
        res = R.research_and_enrich(sid, c)

    assert "bank_iban" not in res["fields"] and "vat_number" not in res["fields"]
    assert "bank_iban" not in res["proposed"] and "vat_number" not in res["proposed"]
    assert "bank_iban" not in _enrichment(sid)["fields"]


# ---------------------------------------------------------------------------
# the human path still works, and is the only thing that writes
# ---------------------------------------------------------------------------
def test_a_human_approval_still_fills_an_empty_field(monkeypatch):
    """The other half of propose-only. If this breaks, the feature is gone
    rather than governed."""
    sid = f"{IDP}7"
    _seed(sid, "Acme Widgets Ltd")
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))
    with get_conn() as c:
        R.research_and_enrich(sid, c)
    eid = _enrichment(sid)["enrichment_id"]

    with get_conn() as c:
        res = R.apply_enrichment(eid, "reviewer1", c)

    assert "website_url" in res["applied"]
    assert _supplier(sid, ["website_url"])["website_url"] == "https://acme.example"


def test_what_was_proposed_is_what_a_human_approval_applies(monkeypatch):
    """A queue that promises a fill it will not perform is worse than no queue.
    One rule set, used by both."""
    sid = f"{IDP}8"
    _seed(sid, "Acme Widgets Ltd", city="Leeds")  # city already set -> not fillable
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))

    with get_conn() as c:
        res = R.research_and_enrich(sid, c)
    proposed = set(res["proposed"])
    eid = _enrichment(sid)["enrichment_id"]

    with get_conn() as c:
        applied = set(R.apply_enrichment(eid, "reviewer1", c)["applied"])

    assert proposed == applied, f"proposed {proposed} but applied {applied}"
    assert "city" not in proposed, "a field that already has a value is not fillable"
    assert _supplier(sid, ["city"])["city"] == "Leeds"


def test_the_write_is_attributed_to_the_reviewer_not_to_the_agent(monkeypatch):
    """`last_modified_by` was hardcoded to 'agentnick_web'. No agent writes
    these columns any more, so that string is now simply false on every row."""
    sid = f"{IDP}9"
    _seed(sid, "Acme Widgets Ltd")
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))
    with get_conn() as c:
        R.research_and_enrich(sid, c)
    eid = _enrichment(sid)["enrichment_id"]

    with get_conn() as c:
        R.apply_enrichment(eid, "reviewer-42", c)

    assert _supplier(sid, ["last_modified_by"])["last_modified_by"] == "reviewer-42"


# ---------------------------------------------------------------------------
# the review queue shows the actual proposal
# ---------------------------------------------------------------------------
def test_the_review_queue_offers_exactly_the_proposed_fields(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routers import supplier_research as SRR

    sid = f"{IDP}A"
    _seed(sid, "Acme Widgets Ltd", city="Leeds")
    monkeypatch.setattr(R, "_run_loop", lambda name: (_CANNED, _EVIDENCE))
    with get_conn() as c:
        res = R.research_and_enrich(sid, c)
    eid = _enrichment(sid)["enrichment_id"]

    app = FastAPI(); app.include_router(SRR.router)
    data = TestClient(app).get("/suppliers/enrichment/reviews?status=pending").json()
    mine = [r for r in data["reviews"] if r["enrichment_id"] == eid]

    assert mine, "a proposal must appear in the review queue"
    assert set(mine[0]["would_fill"]) == set(res["proposed"]), (
        f"the queue offers {mine[0]['would_fill']} but the proposal is "
        f"{sorted(res['proposed'])}")
