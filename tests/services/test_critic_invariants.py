"""The three rules a critique may never break.

Code refuses malformed output rather than repairing it: a critique quietly
corrected is a critique nobody knows was wrong.
"""
from src.services.opportunity_critic.invariants import check_invariants

_CLEAN = {
    "verdict": "VALID",
    "value": {"detector_proposed": 1000.0, "critic_addressable": 800.0},
    "gaps": [{"gap_id": "G1", "blocking": False}],
}


def test_a_clean_critique_has_no_violations():
    assert check_invariants(_CLEAN) == []


def test_valid_may_never_carry_a_value_above_the_detectors():
    bad = {"verdict": "VALID",
           "value": {"detector_proposed": 1000.0, "critic_addressable": 1500.0},
           "gaps": []}
    violations = check_invariants(bad)
    assert any("above the detector" in v for v in violations)


def test_valid_at_exactly_the_detectors_value_is_allowed():
    ok = {"verdict": "VALID",
          "value": {"detector_proposed": 1000.0, "critic_addressable": 1000.0},
          "gaps": []}
    assert check_invariants(ok) == []


def test_valid_may_never_carry_a_blocking_gap():
    bad = {"verdict": "VALID",
           "value": {"detector_proposed": 1000.0, "critic_addressable": 800.0},
           "gaps": [{"gap_id": "G1", "blocking": True}]}
    violations = check_invariants(bad)
    assert any("blocking gap" in v for v in violations)


def test_unassessed_must_carry_at_least_one_blocking_gap():
    bad = {"verdict": "UNASSESSED",
           "value": {"detector_proposed": 1000.0, "critic_addressable": None},
           "gaps": [{"gap_id": "G1", "blocking": False}]}
    violations = check_invariants(bad)
    assert any("at least one blocking gap" in v for v in violations)


def test_valid_reframed_is_held_to_the_value_ceiling_too():
    bad = {"verdict": "VALID_REFRAMED",
           "value": {"detector_proposed": 1000.0, "critic_addressable": 1500.0},
           "gaps": []}
    assert any("above the detector" in v for v in check_invariants(bad))


def test_an_unknown_verdict_is_a_violation():
    assert check_invariants({"verdict": "PROBABLY_FINE", "value": {}, "gaps": []})
