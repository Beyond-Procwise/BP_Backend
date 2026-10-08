"""A claim is confirmed through the same endpoint-level mechanism as any other assumption, and that is what makes the draft ready. Real Postgres."""

import json

import pytest

from src.services.draft_assurance import capture
from tests.email_evals.test_learning import db  # noqa: F401  (fixture)


@pytest.fixture
def clean(db):
    with db.cursor() as cur:
        cur.execute("TRUNCATE email_agent.bp_draft_outcome, email_agent.bp_draft_capture RESTART IDENTITY CASCADE")
    return db


def claimed_draft(conn, uid="U-CLAIM"):
    facts = {"supplier_current_offer": {"value": "47.5000", "label": "Supplier's latest offer", "source": "postgres", "table": "supplier_response",
                                         "column": "price", "row_id": "2", "retrieved_at": "t", "claim": True, "origin": "extracted_unverified"}}
    items = [{"id": "claim:supplier_current_offer", "key": "claim.supplier_current_offer", "resolution": None,
              "text": "Supplier's latest offer was read from the supplier's email by software and has not been confirmed by a person."}]
    a = {"family_id": "negotiation_counter", "family_version": 1, "mode": "shadow", "status": "needs_review", "facts": facts, "conflicts": [],
         "reasoned": {}, "assumptions": [], "violations": [], "repaired": False, "carried_unverified": {}, "unverified_figures": [],
         "assumption_items": items, "family_source": "declared", "ready": False, "clarification": {},
         "accountability": {"initiated_by": "NegotiationAgent", "kind": "agent"}}
    assert capture.record_draft(conn, {"unique_id": uid, "workflow_id": "wf", "supplier_id": "S-1", "body": "<p>Hello</p>", "metadata": {}, "assurance": a})
    return uid


def test_a_draft_resting_on_a_claim_is_not_ready_until_a_person_confirms_it(clean):
    uid = claimed_draft(clean)
    assert capture.readiness(clean, uid)["ready"] is False
    result = capture.confirm_assumptions(clean, uid, [{"id": "claim:supplier_current_offer", "action": "confirm"}], "nick")
    assert result["ok"] is True and capture.readiness(clean, uid)["ready"] is True


def test_who_confirmed_the_claim_is_recorded_as_that_person(clean):
    uid = claimed_draft(clean)
    capture.confirm_assumptions(clean, uid, [{"id": "claim:supplier_current_offer", "action": "confirm"}], "sub-nick")
    raw = capture.load_raw(clean, uid)
    assert raw["assumptions_resolution"]["claim:supplier_current_offer"]["by"] == "sub-nick"


def test_the_reviewer_view_of_a_claimed_draft_says_so_in_words(clean):
    uid = claimed_draft(clean)
    view = capture.to_view(capture.load_raw(clean, uid))
    offer = view["facts"]["supplier_current_offer"]
    assert offer["claim"] is True and "email" in offer["origin_label"].lower() and view["ready"] is False
    assert [i["id"] for i in view["assumptions"]] == ["claim:supplier_current_offer"]
    blob = json.dumps(view)
    assert "supplier_response" not in blob and "extraction_status" not in blob


def test_confirming_a_claim_does_not_touch_the_product_row(clean):
    """Confirming for a draft is a judgement about THIS draft. Marking the supplier_response row confirmed would be a product-table write
    the writer role cannot make, and no endpoint does."""
    uid = claimed_draft(clean)
    with clean.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.supplier_response WHERE extraction_status = 'confirmed'")
        before = cur.fetchone()[0]
    capture.confirm_assumptions(clean, uid, [{"id": "claim:supplier_current_offer", "action": "confirm"}], "nick")
    with clean.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.supplier_response WHERE extraction_status = 'confirmed'")
        assert cur.fetchone()[0] == before
