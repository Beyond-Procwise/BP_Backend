"""Persistence for critiques and their gaps.

The load-bearing behaviour is the return value: record_critique returns None
when it wrote nothing, and the caller must not suppress a finding on a None.
"""
from unittest.mock import MagicMock, patch

from src.services.opportunity_critic.store import record_critique

_CRITIQUE = {
    "opportunity_ref_id": "ref-1",
    "detector_type": "Price Benchmark Variance",
    "verdict": "INVALID",
    "confidence": "UNASSESSED",
    "original_claim": "Paying 10% above benchmark",
    "critic_claim": None,
    "negotiator_note": "The benchmark is the cheapest price ever recorded.",
    "value": {"detector_proposed": 18330.80, "critic_addressable": None,
              "currency": "GBP", "basis": "annualised", "haircuts_applied": []},
    "lever": {"exists": False, "type": "none"},
    "tests": [{"test": "anchor_validity", "result": "INVALIDATE", "reason": "..."},
              {"test": "materiality", "result": "PASS", "reason": "..."}],
    "gaps": [
        {"gap_id": "G1", "test": "anchor_validity", "type": "DETECTOR_LOGIC",
         "what_is_missing": "Anchor selection uses .min() over avg_price",
         "why_it_matters": "Every variance from it is measured off a floor",
         "blocking": False, "resolves_to": "prevents the class of error",
         "likely_source": "detector fix", "owner_hint": "engineering",
         "effort": "LOW"},
    ],
    "prompt_version": 1,
    "policy_versions": {"opportunity_critic_thresholds": 1},
    "formula_versions": {"critic.annualised_rate": "1.0.0+abc123"},
    "run_id": "wf-9",
}


def _fake_conn():
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (77,)
    conn.cursor.return_value = cur
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    return conn, cur


def test_returns_the_new_critique_id():
    conn, _ = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        assert record_critique(_CRITIQUE, shadowed=False) == 77


def test_every_test_result_is_written_including_passes():
    # "anchor_validity PASS" is the sentence that defends a finding in the
    # room, so a critique that stored only the tests that fired would be
    # useless to a negotiator.
    conn, cur = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=False)
    written = [c for c in cur.execute.call_args_list
               if "bp_opportunity_critique" in c.args[0]][0]
    tests_blob = next(p for p in written.args[1]
                      if isinstance(p, str) and "anchor_validity" in p)
    assert "materiality" in tests_blob, "a PASS result was dropped"
    assert '"result": "PASS"' in tests_blob


def test_gaps_are_written_as_rows():
    conn, cur = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=False)
    assert any("bp_opportunity_gap" in c.args[0] for c in cur.execute.call_args_list)


def test_a_failed_write_returns_none_so_nothing_gets_suppressed():
    conn, cur = _fake_conn()
    cur.execute.side_effect = RuntimeError("connection lost")
    with patch("src.services.db.get_conn", return_value=conn):
        assert record_critique(_CRITIQUE, shadowed=False) is None


def test_a_failed_write_never_raises():
    # Bookkeeping must not break the pipeline, exactly as
    # policy_observation.record does not.
    conn, cur = _fake_conn()
    cur.execute.side_effect = RuntimeError("connection lost")
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=True)  # must not raise


def test_versions_that_produced_the_verdict_are_persisted():
    conn, cur = _fake_conn()
    with patch("src.services.db.get_conn", return_value=conn):
        record_critique(_CRITIQUE, shadowed=False)
    written = [c for c in cur.execute.call_args_list
               if "bp_opportunity_critique" in c.args[0]][0]
    assert any("critic.annualised_rate" in str(p) for p in written.args[1])
