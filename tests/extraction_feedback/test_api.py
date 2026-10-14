"""T7: review API — list / approve / reject."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import extraction_feedback
from src.services.db import get_conn


@pytest.fixture
def client():
    from api.auth import require_user

    app = FastAPI()
    app.include_router(extraction_feedback.router)
    # The review endpoints resolve the caller (P8 phase 2). These tests are
    # about the review flow, not authentication, and require_user's global
    # mode is test-order-sensitive, so the principal is pinned: nobody.
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app)


def _insert_pending(vendor, dedup, doc_type="invoice", field="tax_amount", hint="capture VAT as tax_amount"):
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.bp_extraction_hint_proposal "
                "(doc_type, vendor_key, field_name, dedup_key, evidence, proposed_hint, status) "
                "VALUES (%s,%s,%s,%s,%s::jsonb,%s,'pending') RETURNING proposal_id",
                (doc_type, vendor, field, dedup, json.dumps({"sample_count": 4}), hint),
            )
            pid = cur.fetchone()[0]
        c.commit()
    return pid


def test_list_and_approve(client, cleanup_hint_names):
    names, dedups = cleanup_hint_names
    names.append("vhint::invoice::APITESTCO::tax_amount")
    dedups.append("t-api-approve")
    pid = _insert_pending("APITESTCO", "t-api-approve")

    listing = client.get("/extraction/proposals?status=pending").json()
    assert any(p["proposal_id"] == pid for p in listing["proposals"])

    res = client.post(f"/extraction/proposals/{pid}/approve", json={"approver": "tester"})
    assert res.status_code == 200
    assert res.json()["version"] == 1

    # second approve → 409 (already approved)
    assert client.post(f"/extraction/proposals/{pid}/approve", json={"approver": "tester"}).status_code == 409


def test_reject(client, cleanup_hint_names):
    names, dedups = cleanup_hint_names
    dedups.append("t-api-reject")
    pid = _insert_pending("APITESTCO2", "t-api-reject")

    res = client.post(f"/extraction/proposals/{pid}/reject", json={"approver": "tester", "reason": "noise"})
    assert res.status_code == 200 and res.json()["status"] == "rejected"

    detail = client.get(f"/extraction/proposals/{pid}").json()
    assert detail["status"] == "rejected" and detail["review_reason"] == "noise"
