"""The four cases the resolution layer exists to get right, from real behaviour."""
import math

from src.services.resolution import (
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    ResourceCapacity,
    resolve,
)

N_TO_1 = CardinalityRule("invoice_po", "N:1", None, 1)
ONE_TO_1 = CardinalityRule("invoice_po", "1:1", 1, 1)


def _req(edges, rules, capacities=(), request_id="R"):
    return ResolutionRequest(
        request_id=request_id,
        edges=tuple(edges),
        capacities=tuple(capacities),
        rules=tuple(rules),
        profile_registry_version="test-v1",
    )


def _links(result):
    return sorted((l.source_id, l.target_id) for l in result.links)


def test_four_invoices_summing_exactly_to_one_po_resolve_as_a_set():
    """Each invoice bills a quarter of the PO. All four belong; the PO is exactly
    consumed; and dropping any one of them costs the whole unassignment penalty,
    so the set is decisive, not a coincidence."""
    edges = [
        CandidateEdge(f"INV-{i}", "PO-1", 3.0, 0.95, "invoice_po", {"po:PO-1": 250.0})
        for i in range(1, 5)
    ]
    result = resolve(_req(edges, [N_TO_1], [ResourceCapacity("po:PO-1", 1000.0, 0.0)]))

    assert result.status == "RESOLVED"
    assert _links(result) == [(f"INV-{i}", "PO-1") for i in range(1, 5)]
    assert result.unassigned_sources == ()
    assert all(l.margin > 50.0 for l in result.links)


def test_five_invoices_where_any_four_fit_are_degenerate():
    """The PO holds four of the five. Which four is arbitrary, and a layer that
    reported that as a confident answer would be lying."""
    edges = [
        CandidateEdge(f"INV-{i}", "PO-1", 3.0, 0.95, "invoice_po", {"po:PO-1": 250.0})
        for i in range(1, 6)
    ]
    result = resolve(_req(edges, [N_TO_1], [ResourceCapacity("po:PO-1", 1000.0, 0.0)]))

    assert result.status == "DEGENERATE"
    assert len(result.links) == 4
    assert len(result.unassigned_sources) == 1
    assert all(l.margin == 0.0 for l in result.links)


def test_the_optimal_pair_beats_the_per_source_best_choices():
    """Greedy takes S1's favourite and leaves S2 with nothing good. Taken as a
    set, crossing them over carries far more evidence."""
    edges = [
        CandidateEdge("S1", "T1", 5.0, 0.99, "invoice_po", {}),
        CandidateEdge("S1", "T2", 4.0, 0.98, "invoice_po", {}),
        CandidateEdge("S2", "T1", 4.9, 0.99, "invoice_po", {}),
        CandidateEdge("S2", "T2", 0.0, 0.50, "invoice_po", {}),
    ]
    result = resolve(_req(edges, [ONE_TO_1]))

    assert _links(result) == [("S1", "T2"), ("S2", "T1")]
    assert result.objective == -8.9


def test_elimination_assigns_a_mediocre_link_when_it_is_the_only_one_left():
    """0.68 against its only feasible PO. The alternative is not a better PO, it
    is no PO at all — so the link holds, and the margin says the alternative was
    leaving the invoice unplaced."""
    logodds_068 = math.log(0.68 / 0.32)
    edges = [
        CandidateEdge("INV-1", "PO-1", logodds_068, 0.68, "invoice_po",
                      {"po:PO-1": 100.0}),
        CandidateEdge("INV-1", "PO-2", 2.0, 0.88, "invoice_po", {"po:PO-2": 100.0}),
    ]
    capacities = [
        ResourceCapacity("po:PO-1", 100.0, 0.0),
        ResourceCapacity("po:PO-2", 0.0, 0.0),   # already fully consumed
    ]
    result = resolve(_req(edges, [N_TO_1], capacities))

    assert _links(result) == [("INV-1", "PO-1")]
    assert result.links[0].displaced_by == ()
    assert result.links[0].margin > 100.0


def test_no_resource_is_consumed_beyond_capacity_plus_tolerance():
    """Two invoices of 800 against a 1000 PO: only one can hold."""
    edges = [
        CandidateEdge("INV-1", "PO-1", 3.0, 0.95, "invoice_po", {"po:PO-1": 800.0}),
        CandidateEdge("INV-2", "PO-1", 2.0, 0.88, "invoice_po", {"po:PO-1": 800.0}),
    ]
    result = resolve(_req(edges, [N_TO_1], [ResourceCapacity("po:PO-1", 1000.0, 20.0)]))

    assert _links(result) == [("INV-1", "PO-1")]
    assert result.unassigned_sources == ("INV-2",)
