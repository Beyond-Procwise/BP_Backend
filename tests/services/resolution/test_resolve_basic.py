"""Phase 1 resolution layer: the smallest behaviours the entry point must have."""
from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    resolve,
)


def _req(edges, rules=(), capacities=(), request_id="R1"):
    return ResolutionRequest(
        request_id=request_id,
        edges=tuple(edges),
        capacities=tuple(capacities),
        rules=tuple(rules),
        profile_registry_version="test-v1",
    )


def test_single_edge_is_resolved():
    req = _req(
        [CandidateEdge("INV-1", "PO-1", log_odds=2.0, confidence=0.88,
                       profile_id="invoice_po", consumes={})],
        rules=[CardinalityRule("invoice_po", "N:1", None, 1)],
    )

    result = resolve(req)

    assert result.status == "RESOLVED"
    assert [(l.source_id, l.target_id) for l in result.links] == [("INV-1", "PO-1")]
    assert result.unassigned_sources == ()
    assert result.request_id == "R1"


def _one_to_one(profile="invoice_po"):
    return CardinalityRule(profile, "1:1", 1, 1)


def test_a_decisive_assignment_reports_a_positive_margin():
    """INV-1 fits PO-1 far better than PO-2, so forbidding the winning link costs
    real evidence — that cost is the margin."""
    req = _req(
        [
            CandidateEdge("INV-1", "PO-1", 4.0, 0.98, "invoice_po", {}),
            CandidateEdge("INV-1", "PO-2", 0.5, 0.62, "invoice_po", {}),
        ],
        rules=[_one_to_one()],
    )

    result = resolve(req)

    assert [(l.source_id, l.target_id) for l in result.links] == [("INV-1", "PO-1")]
    link = result.links[0]
    assert link.margin == 3.5          # -0.5 (second best) minus -4.0 (best)
    assert link.margin_normalised > 0.5
    assert link.displaced_by == ("PO-2",)
    assert result.status == "RESOLVED"


def test_an_exact_tie_is_degenerate():
    """Two targets the source matches identically well. The solver must still
    return one, but the evidence did not choose it."""
    req = _req(
        [
            CandidateEdge("INV-1", "PO-1", 3.0, 0.95, "invoice_po", {}),
            CandidateEdge("INV-1", "PO-2", 3.0, 0.95, "invoice_po", {}),
        ],
        rules=[_one_to_one()],
    )

    result = resolve(req)

    assert result.status == "DEGENERATE"
    assert len(result.links) == 1
    assert result.links[0].margin == 0.0
