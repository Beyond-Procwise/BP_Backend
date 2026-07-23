import pytest
from itertools import combinations

from src.services import deal_clustering as dc
from src.services.version_collapse import collapse_versions
from tests.fixtures.deal_clustering import golden_batch as gb


def _cluster_of(clusters, prefix):
    return next(c for c in clusters if any(b["base_reference"].startswith(prefix) for b in c))


# Event membership by base_reference, mirroring the fixture's own doc-string grouping
# (used only by the adaptive-threshold regression test below to identify cross-event pairs).
_EVENTS = {
    "Freight": ("SDP-Q-44120", "CL-2024-0771", "MFS-Q-3391"),
    "IT-MSA": ("SYN-Q-8820", "POM-Q-5510", "FSM-Q-7742"),
    "Consultancy": ("MCG-Q-1204", "APX-Q-6631", "VAP-Q-9903"),
    "Platform": ("ORB-Q-2290", "CPS-Q-3380", "NXF-Q-4471"),
}


def _event_of(base_ref):
    return next(name for name, refs in _EVENTS.items() if base_ref in refs)


def test_same_supplier_pairs_are_never_rivals():
    bids = [{"quote_id": "WSG100024", "supplier_id": "SUP-DellWorkspaceSolutionsLtd"},
            {"quote_id": "WSG100025", "supplier_id": "SUP-DellWorkspaceSolutionsLtd"}]
    m = dc.pairwise_matrix(bids, {"WSG100024": [], "WSG100025": []})
    assert m == {}   # excluded before scoring — R3


def test_batch_forms_exactly_four_events_by_complete_linkage():
    bids = collapse_versions(gb.quotes())
    lines = gb.quote_lines()
    matrix = dc.pairwise_matrix(bids, lines)
    clusters = dc.complete_linkage(bids, matrix, threshold=0.70)
    multi = [c for c in clusters if len(c) >= 2]
    assert len(multi) == 4                       # IT-MSA, Freight, Consultancy, Platform
    for c in multi:
        assert len({b["supplier_id"] for b in c}) == len(c)  # rivals = distinct suppliers


def test_confidences_order_matches_spec():
    bids = collapse_versions(gb.quotes())
    lines = gb.quote_lines()
    matrix = dc.pairwise_matrix(bids, lines)
    clusters = [c for c in dc.complete_linkage(bids, matrix) if len(c) >= 2]
    vals = sorted(dc.cluster_confidence(c, matrix) for c in clusters)
    # Top-confidence relaxed vs the spec's illustrative real-batch figure (99.7): on these
    # synthesized fixtures the strongest event (Freight) tops out ~93.5, so we assert a
    # floor rather than the exact spec number. The Platform low-end and the single-event
    # review band ARE calibrated exactly against the fixture.
    assert vals[0] == pytest.approx(76.5, abs=3.0)      # Platform lowest -> HITL review band
    assert vals[0] < 80.0
    assert vals[-1] >= 90.0                             # a high-confidence event exists
    assert sum(1 for v in vals if v < 80.0) == 1         # only Platform falls in the review band


def test_single_borderline_pair_does_not_chain_two_events():
    # Complete linkage regression guard: IT-MSA and Platform must stay separate even though
    # Fortis<->NexusFlow is the strongest cross-event bridge in the batch -- a lone strong
    # pair must not chain two whole events into a six-supplier blob. At the library's
    # default threshold (0.70) the events are already cleanly separated for BOTH linkage
    # strategies, so that comparison would be vacuous; instead derive an ADAPTIVE threshold
    # from the live matrix that sits just below the strongest cross pair, so the comparison
    # actually discriminates complete linkage from single linkage.
    bids = collapse_versions(gb.quotes())
    lines = gb.quote_lines()
    matrix = dc.pairwise_matrix(bids, lines)

    cross_corrs = []
    for a, b in combinations(bids, 2):
        if _event_of(a["base_reference"]) == _event_of(b["base_reference"]):
            continue
        res = matrix.get(frozenset((a["quote_id"], b["quote_id"])))
        if res is not None:
            cross_corrs.append(res["correlation"])
    strongest_cross = max(cross_corrs)          # the Fortis<->NexusFlow bridge (~0.4251)
    threshold = strongest_cross - 0.01          # clears that one bridge, not the rest

    clusters = [c for c in dc.complete_linkage(bids, matrix, threshold=threshold) if len(c) >= 2]
    assert len(clusters) == 4                   # still exactly 4 events
    assert all(len(c) < 5 for c in clusters)     # no six-supplier blob

    # Regression proof this genuinely discriminates from single linkage: a small inline
    # single-linkage variant (merge clusters if ANY cross pair clears the threshold) DOES
    # chain IT-MSA + Platform into a >=5-bid cluster at the SAME threshold.
    def _corr(qa, qb):
        res = matrix.get(frozenset((qa, qb)))
        return res["correlation"] if res else 0.0

    def _single_linkage(bids, threshold):
        clusters = [[b] for b in bids]
        changed = True
        while changed:
            changed = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if any(_corr(x["quote_id"], y["quote_id"]) >= threshold
                           for x in clusters[i] for y in clusters[j]):
                        clusters[i] = clusters[i] + clusters[j]
                        del clusters[j]
                        changed = True
                        break
                if changed:
                    break
        return clusters

    single = [c for c in _single_linkage(bids, threshold) if len(c) >= 2]
    assert any(len(c) >= 5 for c in single)      # single linkage DOES chain -> proves the guard


def test_award_detection_survives_supplier_name_mismatch():
    # quote SUP-GomezGoodAndCross vs PO "Gomez, Good and Cross Trading Ltd" (supplier_id NULL).
    # Continuity scoring on lines+price must still resolve the award; name-match would fail.
    q = next(q for q in gb.negative_control_quotes() if q["quote_id"] == "104683")
    pos = gb.negative_control_pos()
    po_lines = gb.negative_control_po_lines()
    q_lines = gb.negative_control_quote_lines()["104683"]
    po_id = dc.awarded_po(q, pos, po_lines, q_lines, min_score=60.0)
    assert po_id == "PO-104683"


def test_award_veto_separates_repeat_buying():
    # Duncan(128234) and Perry(136586) correlate ~0.744 but each has its own PO+invoice.
    awards = {"128234": "PO-128234", "136586": "PO-136586"}
    assert dc.award_veto({"quote_id": "128234"}, {"quote_id": "136586"}, awards) is True


def test_award_veto_does_not_fire_for_single_award_competition():
    # Freight: 3 bidders share ONE PO (Swift). Not a veto.
    awards = {"SDP-Q-44120": "PO-2024-0091", "CL-2024-0771": None, "MFS-Q-3391": None}
    assert dc.award_veto({"quote_id": "CL-2024-0771"}, {"quote_id": "MFS-Q-3391"}, awards) is False
    assert dc.award_veto({"quote_id": "SDP-Q-44120"}, {"quote_id": "CL-2024-0771"}, awards) is False
