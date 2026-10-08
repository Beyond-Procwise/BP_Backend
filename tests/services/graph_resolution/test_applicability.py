"""Optional signals join a pair only when both documents can supply them.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_applicability.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                  # noqa: E402
from src.services.graph_resolution import applicability as ap                  # noqa: E402

le.register_signal("tap_ref", lambda s, t, a, b: (1.0, "OK") if s.get("ref") == t.get("id") else (0.5, "MISSING"))
le.register_signal("tap_opt", lambda s, t, a, b:
                   (0.5, "MISSING") if not (s.get("x") and t.get("x"))
                   else ((1.0, "OK") if s["x"] == t["x"] else (0.0, "CONFLICT")))

BASE = [{"id": "ref", "cluster": "reference", "tier": 1, "weight": 5, "appl": 1.0,
         "cap": 0.45, "kind": "tap_ref", "reads": ["ref"]}]
OPT = [{"id": "opt", "cluster": "terms", "tier": 3, "weight": 2, "appl": 1.0,
        "cap": 0.90, "kind": "tap_opt", "reads": ["x"]}]
PARAMS = {"p0": 0.02, "alpha": 0.35, "floor": 0.55, "date_field": "d"}


def _direct_base_score(src, tgt):
    le.register_profile("tap_base_only", {**PARAMS, "signals": BASE})
    return le.score_link(src, tgt, "tap_base_only")


def _base(name):
    le.register_profile(name, {**PARAMS, "signals": BASE})
    return name


def test_absent_optional_data_scores_exactly_as_the_base_profile():
    src, tgt = {"id": "C", "ref": "P"}, {"id": "P"}
    got = ap.score_pair(_base("tap_prof"), BASE, OPT, src, tgt)
    assert got["F"] == _direct_base_score(src, tgt)["F"]
    assert got["profile"] == "tap_prof"
    assert "tap_prof" in le.PROFILES and [s["id"] for s in le.PROFILES["tap_prof"]["signals"]] == ["ref"]


def test_present_optional_data_joins_the_pair_and_is_reported_under_the_base_name():
    src, tgt = {"id": "C", "ref": "P", "x": "GBP"}, {"id": "P", "x": "GBP"}
    got = ap.score_pair(_base("tap_prof2"), BASE, OPT, src, tgt)
    assert [s["id"] for s in got["signals"]] == ["ref", "opt"]
    assert got["profile"] == "tap_prof2"
    assert "tap_prof2+opt" in le.PROFILES


def test_a_conflicting_optional_signal_lowers_the_score_below_the_base():
    src, tgt = {"id": "C", "ref": "P", "x": "GBP"}, {"id": "P", "x": "USD"}
    got = ap.score_pair(_base("tap_prof3"), BASE, OPT, src, tgt)
    assert got["F"] < _direct_base_score(src, tgt)["F"]


def test_a_variant_follows_its_base_profile_when_the_base_is_recalibrated():
    """scripts/graph_resolution/calibrate.py tunes a profile by changing its p0/alpha.
    A variant that kept the copy taken at first use would go on scoring with the old
    values, silently, for every pair that carries an optional signal."""
    src, tgt = {"id": "C", "ref": "P", "x": "GBP"}, {"id": "P", "x": "GBP"}
    name = _base("tap_prof4")
    before = ap.score_pair(name, BASE, OPT, src, tgt)["F"]
    le.PROFILES[name]["alpha"] = 0.10                      # a recalibration
    after = ap.score_pair(name, BASE, OPT, src, tgt)["F"]
    assert after != before
    assert le.PROFILES[name + "+opt"]["alpha"] == 0.10


def test_an_unregistered_base_profile_is_refused():
    import pytest
    with pytest.raises(KeyError):
        ap.score_pair("tap_never_registered", BASE, OPT, {"id": "C"}, {"id": "P"})


def test_the_variant_name_does_not_depend_on_declaration_order():
    a = {"id": "b"}, {"id": "a"}
    assert ap.variant_name("p", list(a)) == ap.variant_name("p", list(reversed(a))) == "p+a+b"
    assert ap.variant_name("p", []) == "p"
