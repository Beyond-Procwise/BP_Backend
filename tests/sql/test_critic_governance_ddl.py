"""The governance seed carries the critic's rules and prompt, and ships shadow empty."""
from pathlib import Path

SQL = (Path(__file__).resolve().parents[2]
       / "deploy" / "sql" / "2026-09-09_critic_governance.sql").read_text()


def test_system_prompt_is_seeded_as_a_governed_row():
    assert "opportunity_critic_system" in SQL
    assert "proc.bp_prompt" in SQL


def test_every_threshold_is_a_policy_row_not_a_constant():
    for key in ("index_band_pp", "materiality_floor_gbp", "relative_gap_floor",
                "anchor_stale_days", "friction_bands"):
        assert key in SQL


def test_shadow_enrolment_ships_empty():
    # Nothing is suppressed on day one. Enrolling is a governed edit.
    assert '"shadow_detectors": []' in SQL


def test_seeds_are_guarded_so_reruns_are_no_ops():
    assert SQL.count("WHERE NOT EXISTS") >= 2
