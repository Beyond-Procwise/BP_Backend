"""A caller cannot put someone else's name on a write.

P8 phase 1. Of the 87 write/send endpoints in src/api/routers, 57 never resolve
the caller into the handler. All 57 are authenticated -- every router is mounted
behind require_user -- but a handler that never learns WHO cannot attribute what
it writes.

Eight of them went further than "no attribution": they took an identity FROM THE
CALLER and stored it. A reviewer field, a created_by, a user_id -- typed into a
request body and written to the record as though it meant something. Those are
the ones here, because they are the ones where attribution was forgeable rather
than merely absent.

The pattern each is brought to is the one `POST /promotion/review/.../approve`
already used: the subject on the token is the actor, and the field the caller
typed is either kept beside it as an explicitly unverified label (where the
store has somewhere to put one) or not used as identity at all.

These tests assert on what reaches the STORE, not on a response body: the
question is what gets recorded, and a handler can return anything.
"""

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

CALLER = "sub-real-caller"
IMPERSONATED = "sub-someone-else"


class _Principal:
    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"


def _app(router_module, *, subject=CALLER, state=None):
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[router_module.require_user] = (
        (lambda: _Principal(subject)) if subject else (lambda: None)
    )
    for key, value in (state or {}).items():
        setattr(app.state, key, value)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_gates(monkeypatch):
    """Authorization is not what these tests are about; attribution is."""
    for name in ("promotion", "supplier_review", "supplier_research", "analysis",
                 "workflows"):
        try:
            module = __import__(f"api.routers.{name}", fromlist=["*"])
        except Exception:  # noqa: BLE001 - module not importable here, skip it
            continue
        if hasattr(module, "gate"):
            monkeypatch.setattr(module, "gate", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# promotion — a document link, which feeds _stg -> _trgt, the financial record
# ---------------------------------------------------------------------------
def test_confirming_a_parent_link_is_attributed_to_the_token(monkeypatch):
    from api.routers import promotion

    seen = {}
    monkeypatch.setattr(promotion, "confirm_parent_link",
                        lambda *a, **k: seen.update(k) or {"status": "ok"})

    _app(promotion).post(
        "/promotion/link-proposals/invoice/INV-1/confirm",
        json={"po_id": "PO-1", "reviewer": IMPERSONATED},
    )

    assert seen.get("reviewer") == CALLER, (
        f"the caller named the reviewer on a link confirmation: {seen}")


def test_rejecting_a_parent_link_is_attributed_to_the_token(monkeypatch):
    from api.routers import promotion

    seen = {}
    monkeypatch.setattr(promotion, "reject_parent_link",
                        lambda *a, **k: seen.update(k) or {"status": "ok"})

    _app(promotion).post(
        "/promotion/link-proposals/invoice/INV-1/reject",
        json={"po_id": "PO-1", "reviewer": IMPERSONATED},
    )

    assert seen.get("reviewer") == CALLER, seen


# ---------------------------------------------------------------------------
# supplier_review — the supplier master
# ---------------------------------------------------------------------------
def test_confirming_a_supplier_review_is_attributed_to_the_token(monkeypatch):
    from api.routers import supplier_review

    seen = {}
    monkeypatch.setattr(supplier_review.SR, "confirm_review",
                        lambda review_id, reviewer, conn: seen.update(reviewer=reviewer) or {"ok": 1})
    monkeypatch.setattr(supplier_review, "get_conn", _fake_conn)

    _app(supplier_review).post("/suppliers/reviews/1/confirm",
                               json={"reviewer": IMPERSONATED})

    assert seen.get("reviewer") == CALLER, (
        f"a merge into the supplier master was signed by a typed name: {seen}")


def test_rejecting_a_supplier_review_is_attributed_to_the_token(monkeypatch):
    from api.routers import supplier_review

    seen = {}
    monkeypatch.setattr(supplier_review.SR, "reject_review",
                        lambda review_id, reviewer, conn: seen.update(reviewer=reviewer) or {"ok": 1})
    monkeypatch.setattr(supplier_review, "get_conn", _fake_conn)

    _app(supplier_review).post("/suppliers/reviews/1/reject",
                               json={"reviewer": IMPERSONATED})

    assert seen.get("reviewer") == CALLER, seen


# ---------------------------------------------------------------------------
# supplier_research — the enrichment review queue (P5's other half)
# ---------------------------------------------------------------------------
def test_rejecting_an_enrichment_is_attributed_to_the_token(monkeypatch):
    from api.routers import supplier_research

    seen = {}
    monkeypatch.setattr(supplier_research.R, "reject_enrichment",
                        lambda eid, reviewer, conn: seen.update(reviewer=reviewer) or {"ok": 1})
    monkeypatch.setattr(supplier_research, "get_conn", _fake_conn)

    _app(supplier_research).post("/suppliers/enrichment/7/reject",
                                 json={"reviewer": IMPERSONATED})

    assert seen.get("reviewer") == CALLER, seen


# ---------------------------------------------------------------------------
# analysis — proc.bp_analysis, a deal's analysis event
# ---------------------------------------------------------------------------
def test_starting_an_analysis_is_attributed_to_the_token(monkeypatch):
    from api.routers import analysis

    seen = {}
    monkeypatch.setattr(analysis.analysis_store, "start",
                        lambda **k: seen.update(k) or "AN-1")

    _app(analysis).post("/analysis",
                        json={"session_id": "ses-1", "created_by": IMPERSONATED})

    assert seen.get("created_by") == CALLER, (
        f"an analysis event was created in someone else's name: {seen}")


# ---------------------------------------------------------------------------
# workflows — an opportunity rejection, and an attachment on a draft that sends
# ---------------------------------------------------------------------------
def test_rejecting_an_opportunity_is_attributed_to_the_token(monkeypatch):
    from api.routers import workflows

    seen = {}

    def _record(agent_nick, opportunity_id, **k):
        seen.update(k)
        return {"status": "rejected", "updated_on": None}

    monkeypatch.setattr(workflows, "record_opportunity_feedback", _record)

    _app(workflows, state={"agent_nick": object()}).post(
        "/workflows/opportunities/OPP-1/reject",
        json={"reason": "not real", "user_id": IMPERSONATED},
    )

    assert seen.get("user_id") == CALLER, seen


def test_the_rejected_opportunity_keeps_the_typed_value_as_a_label(monkeypatch):
    """The store has a metadata dict, so the thing a person typed is not thrown
    away -- it is recorded as what it is."""
    from api.routers import workflows

    seen = {}

    def _record(agent_nick, opportunity_id, **k):
        seen.update(k)
        return {"status": "rejected", "updated_on": None}

    monkeypatch.setattr(workflows, "record_opportunity_feedback", _record)

    _app(workflows, state={"agent_nick": object()}).post(
        "/workflows/opportunities/OPP-1/reject",
        json={"reason": "not real", "user_id": IMPERSONATED},
    )

    assert (seen.get("metadata") or {}).get("user_id_label") == IMPERSONATED, seen


def test_an_attachment_is_attributed_to_the_token(monkeypatch):
    from api.routers import workflows

    persisted = {}
    monkeypatch.setattr(workflows.draft_rfq_emails_repo, "load_by_unique_id",
                        lambda uid: {"unique_id": uid, "payload": None})
    monkeypatch.setattr(workflows, "_load_draft_attachments", lambda draft: [])
    monkeypatch.setattr(workflows, "_persist_draft_attachments",
                        lambda nick, uid, records: persisted.update(records=records))

    class _Service:
        _ATTACHMENT_MAX_BYTES = 10_000_000
        _ATTACHMENT_MAX_TOTAL_BYTES = 20_000_000

        def __init__(self, agent_nick):
            pass

        def _write_s3_bytes(self, key, data, content_type):
            return None

    monkeypatch.setattr(workflows, "EmailDispatchService", _Service)

    _app(workflows, state={"agent_nick": object()}).post(
        "/workflows/email/UID-1/attachments",
        files={"files": ("quote.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        data={"user_id": IMPERSONATED},
    )

    records = persisted.get("records") or []
    assert records, "nothing was persisted"
    assert records[0]["added_by"] == CALLER, (
        f"an attachment on a draft that will be SENT was filed under a typed "
        f"name: {records[0]}")
    assert records[0].get("added_by_label") == IMPERSONATED, records[0]


# ---------------------------------------------------------------------------
# with no principal there is no actor -- not a fallback to the typed value
# ---------------------------------------------------------------------------
def test_without_a_principal_the_typed_name_is_still_not_the_actor(monkeypatch):
    """ASK_AUTH_MODE=off leaves no principal. The write then names nobody, which
    is honest; falling back to the body would be the forgery taken
    conditionally, which is what this whole surface got wrong the first time."""
    from api.routers import analysis

    seen = {}
    monkeypatch.setattr(analysis.analysis_store, "start",
                        lambda **k: seen.update(k) or "AN-1")

    _app(analysis, subject=None).post(
        "/analysis", json={"session_id": "ses-1", "created_by": IMPERSONATED})

    assert seen.get("created_by") is None, seen


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        raise AssertionError("these tests stub the store; no cursor should be needed")


def _fake_conn():
    return _FakeConn()
