"""The evidence subagent assembles; it never judges and never infers.

Live ground truth this encodes (spec section 6.2): opportunity suppliers are
name-derived slugs (SUP-MeridianSystems12), contract suppliers are coded ids
(S9251), and zero of 308 opportunities resolve to a contract row. The correct
behaviour is to say so, not to fuzzy-match across the two key spaces.
"""
from src.services.opportunity_critic.assemble import assemble_candidate


class _FakeCursor:
    def __init__(self, results):
        self._results = results
        self._last = None

    def execute(self, sql, params=None):
        for key, rows in self._results.items():
            if key in sql:
                self._last = rows
                return
        self._last = []

    def fetchall(self):
        return self._last or []

    def fetchone(self):
        return (self._last or [None])[0]


class _FakeConn:
    def __init__(self, results):
        self._cur = _FakeCursor(results)

    def cursor(self):
        return self._cur


_FINDING = {
    "opportunity_ref_id": "ref-1",
    "detector_type": "Price Benchmark Variance",
    "supplier_id": "SUP-MeridianSystems12",
    "financial_impact_gbp": 18330.80,
    "facts_state": "RESOLVED",
    "source_records": ["PO000967", "INV000967-1"],
    "calculation_details": {
        "actual_price": 4998.53, "benchmark_price": 4540.26,
        "quantity": 40.0, "variance_pct": 0.1009,
    },
}


def test_the_anchor_carries_the_benchmark_price():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["anchor"]["value"] == 4540.26


def test_the_anchor_is_labelled_as_a_cheapest_ever_comparator():
    # Not a prior price and not a market benchmark: it is a .min() over
    # avg_price. The critic cannot test anchor validity without knowing that.
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["anchor"]["kind"] == "cheapest_observed"


def test_an_unresolvable_supplier_produces_absent_contract_context():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["contract_context"] == {}


def test_an_unresolvable_supplier_is_reported_not_guessed():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["evidence"]["contract_resolution"] == "SUPPLIER_NOT_IN_CONTRACT_MASTER"


def test_contract_context_is_populated_when_the_supplier_does_resolve():
    conn = _FakeConn({"bp_contract_master": [
        ("C-1", "2022-03-01", "2028-03-01", "GBP", "United Kingdom", "NET30", None),
    ]})
    out = assemble_candidate(_FINDING, conn)
    assert out["contract_context"]["contract_id"] == "C-1"
    assert out["contract_context"]["currency"] == "GBP"


def test_indeterminate_facts_state_tags_evidence_unassessed():
    finding = dict(_FINDING, facts_state="INDETERMINATE")
    out = assemble_candidate(finding, _FakeConn({}))
    assert out["evidence"]["current_confidence"] == "UNASSESSED"


def test_resolved_facts_state_tags_evidence_asserted_not_corroborated():
    # RESOLVED means numbers were parsed out of a document, not that a second
    # source agreed with them. Nothing here can upgrade evidence.
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["evidence"]["current_confidence"] == "ASSERTED"


def test_no_index_is_reported_as_absent_rather_than_omitted():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    assert out["category_context"]["index_pct"] is None
    assert out["category_context"]["index_source"] == "NONE_AVAILABLE"


def test_the_assembler_never_returns_a_verdict_field():
    out = assemble_candidate(_FINDING, _FakeConn({}))
    for judged in ("verdict", "tests", "critic_claim", "value"):
        assert judged not in out
