"""The routing surface for documents that named no purchase order.

The review queue is an F-score window over held documents, and a document with
no parent has no F, so it has never appeared there. These two endpoints are where
the resolution layer's output reaches a person: what the evidence proposes, how
forced the proposal was, and a way to accept one.
"""
from fastapi.testclient import TestClient

from src.api.main import app
from src.services.link_proposals import LinkProposal

client = TestClient(app)

# See tests/api/test_deal_proposals_endpoints.py: the app holds the bare-imported
# `api.routers.promotion`, so that is the module a monkeypatch has to target.
_MOD = "api.routers.promotion"


def _proposal(doc_pk="INV-1", po_id="PO-1", routing="suggested"):
    return LinkProposal(doc_type="invoice", doc_pk=doc_pk, po_id=po_id, F=73.5,
                        margin=9.2, margin_normalised=1.0, routing=routing,
                        alternatives=("PO-2",), claim=600.0, order_remaining=1000.0,
                        within_order_value=True)


def _run(*proposals, **considered):
    from src.services.link_proposals import ProposalRun
    counts = {"documents": 2, "without_supplier": 0, "supplier_holds_no_order": 0,
              "scored": 2}
    counts.update(considered)
    return ProposalRun(tuple(proposals), counts)


def test_the_queue_reports_each_proposal_with_its_margin(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.propose_parent_links",
                        lambda **kw: _run(_proposal(),
                                          _proposal("INV-2", "PO-3", "contested")))

    r = client.get("/promotion/link-proposals")

    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["items"][0] == {
        "doc_type": "invoice", "doc_pk": "INV-1", "po_id": "PO-1", "F": 73.5,
        "margin": 9.2, "margin_normalised": 1.0, "routing": "suggested",
        "alternatives": ["PO-2"], "claim": 600.0, "order_remaining": 1000.0,
        "within_order_value": True,
    }
    assert body["items"][1]["routing"] == "contested"


def test_a_queue_with_nothing_in_it_still_says_what_was_read(monkeypatch):
    """An empty list and an empty corpus are different facts, and a screen that
    cannot tell them apart reports "nothing to link" when nothing was looked at."""
    monkeypatch.setattr(f"{_MOD}.propose_parent_links",
                        lambda **kw: _run(documents=1966, without_supplier=1,
                                          supplier_holds_no_order=751, scored=1214))

    r = client.get("/promotion/link-proposals")

    assert r.json() == {"count": 0, "items": [],
                        "considered": {"documents": 1966, "without_supplier": 1,
                                       "supplier_holds_no_order": 751, "scored": 1214}}


def test_only_invoices_and_quotes_are_accepted():
    r = client.get("/promotion/link-proposals", params={"doc_type": "purchase_order"})

    assert r.status_code == 400


def test_confirming_a_proposal_reports_what_was_written(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.confirm_parent_link",
                        lambda *a, **kw: {"status": "linked", "doc_type": "invoice",
                                          "doc_pk": "INV-1", "po_id": "PO-1",
                                          "F": 73.5, "margin": 9.2,
                                          "routing": "suggested",
                                          "confirmed_by": "ana"})

    r = client.post("/promotion/link-proposals/invoice/INV-1/confirm",
                    json={"po_id": "PO-1", "reviewer": "ana"})

    assert r.status_code == 200
    assert r.json()["po_id"] == "PO-1"


def test_confirming_without_naming_an_order_is_rejected():
    r = client.post("/promotion/link-proposals/invoice/INV-1/confirm", json={})

    assert r.status_code == 400


def test_a_refused_confirmation_is_a_conflict_not_a_success(monkeypatch):
    """Refused means the document's own state says no — it already names an
    order, or this order was never proposed for it. Returning 200 with a refusal
    buried in the body is how a UI ends up reporting a link that was never made.
    """
    monkeypatch.setattr(f"{_MOD}.confirm_parent_link",
                        lambda *a, **kw: {"status": "refused",
                                          "detail": "PO-9 was not proposed"})

    r = client.post("/promotion/link-proposals/invoice/INV-1/confirm",
                    json={"po_id": "PO-9"})

    assert r.status_code == 409
    assert "not proposed" in r.json()["detail"]


def test_confirming_an_unknown_document_is_a_404(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.confirm_parent_link",
                        lambda *a, **kw: {"status": "not_found", "detail": "no such invoice"})

    r = client.post("/promotion/link-proposals/invoice/NOPE/confirm",
                    json={"po_id": "PO-1"})

    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Saying no
# ---------------------------------------------------------------------------
def test_rejecting_a_proposal_reports_what_was_recorded(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.reject_parent_link",
                        lambda *a, **kw: {"status": "rejected", "doc_type": "invoice",
                                          "doc_pk": "INV-1", "po_id": "PO-1",
                                          "rejected_by": "ana"})

    r = client.post("/promotion/link-proposals/invoice/INV-1/reject",
                    json={"po_id": "PO-1", "reviewer": "ana", "note": "different site"})

    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_rejecting_without_naming_an_order_is_rejected():
    """Rejecting a document rather than a pairing would suppress every order it
    could ever be offered, which is not what the button says."""
    r = client.post("/promotion/link-proposals/invoice/INV-1/reject", json={})

    assert r.status_code == 400


def test_a_refused_rejection_is_a_conflict(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.reject_parent_link",
                        lambda *a, **kw: {"status": "refused",
                                          "detail": "PO-9 was not proposed"})

    r = client.post("/promotion/link-proposals/invoice/INV-1/reject",
                    json={"po_id": "PO-9"})

    assert r.status_code == 409


def test_rejecting_an_unknown_document_is_a_404(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.reject_parent_link",
                        lambda *a, **kw: {"status": "not_found", "detail": "no such invoice"})

    r = client.post("/promotion/link-proposals/invoice/NOPE/reject",
                    json={"po_id": "PO-1"})

    assert r.status_code == 404


def test_the_two_decisions_are_separate_endpoints(monkeypatch):
    """A shared endpoint taking a verb is how a client ends up sending the wrong one.
    Confirming writes a reference; rejecting must never be able to."""
    called = []
    monkeypatch.setattr(f"{_MOD}.confirm_parent_link",
                        lambda *a, **kw: called.append("confirm") or {"status": "linked"})
    monkeypatch.setattr(f"{_MOD}.reject_parent_link",
                        lambda *a, **kw: called.append("reject") or {"status": "rejected"})

    client.post("/promotion/link-proposals/invoice/INV-1/reject", json={"po_id": "PO-1"})

    assert called == ["reject"]
