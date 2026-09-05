"""Infeasibility is a finding, in procurement terms — never a swallowed exception."""
from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    ResourceCapacity,
    resolve,
)

N_TO_1 = CardinalityRule("invoice_po", "N:1", None, 1)


def _req(edges, capacities=(), rules=(N_TO_1,)):
    return ResolutionRequest(
        request_id="R", edges=tuple(edges), capacities=tuple(capacities),
        rules=tuple(rules), profile_registry_version="test-v1",
    )


def test_a_document_that_cannot_be_placed_anywhere_is_infeasible():
    """INV-44 claims more than its only PO line can ever hold. No assignment
    fixes that, so the request fails closed rather than quietly dropping it."""
    edges = [
        CandidateEdge("INV-44", "PO-100", 3.0, 0.95, "invoice_po",
                      {"po_line:PO-100:3": 2100.0}),
    ]
    result = resolve(_req(edges, [ResourceCapacity("po_line:PO-100:3", 1400.0, 20.0)]))

    assert result.status == "INFEASIBLE"
    assert result.infeasibility_certificate
    text = " ".join(result.infeasibility_certificate)
    assert "po_line:PO-100:3" in text
    assert "1400.00" in text and "20.00" in text and "2100.00" in text
    assert "INV-44" in text
    assert "infeasible" not in text.lower()   # domain terms, not solver terms


def test_the_certificate_names_every_claimant_of_the_binding_resource():
    edges = [
        CandidateEdge("INV-44", "PO-100", 3.0, 0.95, "invoice_po",
                      {"po_line:PO-100:3": 2100.0}),
        CandidateEdge("INV-45", "PO-100", 3.0, 0.95, "invoice_po",
                      {"po_line:PO-100:3": 100.0}),
    ]
    result = resolve(_req(edges, [ResourceCapacity("po_line:PO-100:3", 1400.0, 20.0)]))

    text = " ".join(result.infeasibility_certificate)
    assert "INV-44" in text and "INV-45" in text


def test_a_negative_bound_is_infeasible_even_with_nothing_assigned():
    edges = [CandidateEdge("INV-1", "PO-1", 3.0, 0.95, "invoice_po", {"po:PO-1": 10.0})]
    result = resolve(_req(edges, [ResourceCapacity("po:PO-1", -5.0, 0.0)]))

    assert result.status == "INFEASIBLE"
    assert result.infeasibility_certificate


def test_losing_a_competition_for_capacity_is_not_infeasible():
    """Being outbid for a PO's remaining value is an ordinary outcome: the loser
    is reported unassigned, and the request still resolves."""
    edges = [
        CandidateEdge("INV-1", "PO-1", 3.0, 0.95, "invoice_po", {"po:PO-1": 800.0}),
        CandidateEdge("INV-2", "PO-1", 2.0, 0.88, "invoice_po", {"po:PO-1": 800.0}),
    ]
    result = resolve(_req(edges, [ResourceCapacity("po:PO-1", 1000.0, 0.0)]))

    assert result.status in ("RESOLVED", "DEGENERATE")
    assert result.infeasibility_certificate is None
    assert result.unassigned_sources == ("INV-2",)
