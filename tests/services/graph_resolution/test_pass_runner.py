from src.services.graph_resolution.pass_runner import (
    to_candidate_edges, band_for_resolution,
)
from src.services.resolution import CandidateEdge


def test_candidate_edges_carry_log_odds_not_F():
    scored = [{"source_id": "SUP-A", "target_id": "S1",
               "result": {"L": 3.2, "L_evidence": 7.1, "P_raw": 0.96, "F": 94.0}}]
    edges = to_candidate_edges(scored, profile_id="supplier_identity")
    assert isinstance(edges[0], CandidateEdge)
    assert edges[0].log_odds == 3.2   # L, prior included: the solver wants full log-odds
    assert edges[0].confidence == 0.96


def test_degenerate_resolution_is_capped_below_auto_link():
    assert band_for_resolution("auto_link", "DEGENERATE") == "review"


def test_resolved_status_leaves_the_band_alone():
    assert band_for_resolution("auto_link", "RESOLVED") == "auto_link"


def test_infeasible_never_yields_an_actionable_band():
    assert band_for_resolution("auto_link", "INFEASIBLE") == "weak_relation"
