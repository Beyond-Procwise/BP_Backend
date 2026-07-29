"""Stable identity for mined opportunities, and retiring what is no longer found.

proc.bp_opportunity was keyed on opportunity_id, which the miner assigns from a
per-run counter during candidate evaluation. The same finding therefore got a
different id on every run, so re-running mining inserted duplicates instead of
updating, nothing was ever retired, and a colliding id could overwrite an
unrelated finding and inherit its lifecycle stage.

Observed live: after the FX fix corrected the figures, the superseded finding
(Ashcroft, GBP 8,423) sat on the Opportunities screen beside the corrected ones
because the new run had written new rows under new ids.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.services import opportunity_store as store


class _Cur:
    """Records the SQL issued, so the identity and scoping rules can be asserted."""

    def __init__(self, rows=None):
        self.sql = []
        self.params = []
        self._rows = rows or []
        self.rowcount = len(self._rows)

    def execute(self, sql, params=None):
        self.sql.append(" ".join(str(sql).split()))
        self.params.append(params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_upsert_is_keyed_on_the_stable_reference_not_the_run_counter():
    cur = _Cur()
    store.upsert_opportunity(cur, {
        "opportunity_id": "12123",
        "opportunity_ref_id": "pol_det_abc123_supplier_item",
        "detector_type": "Price Benchmark Variance",
        "financial_impact_gbp": 100.0,
    })

    sql = cur.sql[0].lower()
    assert "on conflict (opportunity_ref_id)" in sql
    assert "on conflict (opportunity_id)" not in sql


def test_re_detection_clears_a_previous_retirement():
    """A finding that comes back is live again, not left marked as retired."""
    cur = _Cur()
    store.upsert_opportunity(cur, {
        "opportunity_id": "1",
        "opportunity_ref_id": "ref-1",
        "financial_impact_gbp": 10.0,
    })

    assert "retired_at=null" in cur.sql[0].lower().replace(" ", "")


def test_a_progressed_stage_is_still_never_demoted():
    cur = _Cur()
    store.upsert_opportunity(cur, {
        "opportunity_id": "1", "opportunity_ref_id": "ref-1",
        "financial_impact_gbp": 10.0,
    })

    sql = cur.sql[0].lower()
    assert "else proc.bp_opportunity.stage end" in sql


def test_deal_id_is_persisted_when_the_finding_carries_one():
    """A deal-scoped detector (e.g. Invoice Overbilling) knows its deal_id
    directly — it must not depend on the fragile quote-anchored backfill in
    opportunity_linkage.py, which only fires when o.quote_id matches a quote
    document and leaves everything else deal_id=NULL forever."""
    cur = _Cur()
    store.upsert_opportunity(cur, {
        "opportunity_id": "1", "opportunity_ref_id": "ref-1",
        "financial_impact_gbp": 10.0,
        "deal_id": "DEALV2-005049",
    })

    sql = cur.sql[0].lower()
    assert "deal_id" in sql
    assert "DEALV2-005049" in cur.params[0]


def test_deal_id_falls_back_to_calculation_details():
    cur = _Cur()
    store.upsert_opportunity(cur, {
        "opportunity_id": "1", "opportunity_ref_id": "ref-1",
        "financial_impact_gbp": 10.0,
        "calculation_details": {"deal_id": "DEALV2-000001"},
    })

    assert "DEALV2-000001" in cur.params[0]


# ---------------------------------------------------------------------------
# Retirement
# ---------------------------------------------------------------------------

def test_retire_only_touches_identified_rows():
    cur = _Cur()
    store.retire_missing(cur, seen_ref_ids=["ref-a"],
                         detector_types=["Price Benchmark Variance"])

    sql = cur.sql[0].lower()
    assert "stage='identified'" in sql.replace(" = ", "=")
    assert "retired_at=now()" in sql.replace(" = ", "=")
    assert "stage='closed'" in sql.replace(" = ", "=")


def test_retire_is_scoped_to_the_detectors_this_run_produced():
    """A detector that returned nothing is indistinguishable from a broken one
    (proc.contracts does not exist in this corpus), so its findings are left."""
    cur = _Cur()
    store.retire_missing(cur, seen_ref_ids=["ref-a"],
                         detector_types=["Price Benchmark Variance"])

    assert "detector_type" in cur.sql[0].lower()
    params = cur.params[0]
    assert ["Price Benchmark Variance"] in [list(p) if isinstance(p, (list, tuple)) else p
                                            for p in params]


def test_retire_does_nothing_without_detectors():
    cur = _Cur()
    n = store.retire_missing(cur, seen_ref_ids=[], detector_types=[])

    assert n == 0
    assert cur.sql == [], "a run that produced no findings must retire nothing"


def test_retire_spares_findings_below_this_runs_threshold():
    """An ad-hoc high-threshold scan must not delete live low-value findings.

    min_financial_impact filters the run's OWN output, so a finding below it was
    never a candidate for re-detection — its absence means nothing.
    """
    cur = _Cur()
    store.retire_missing(cur, seen_ref_ids=["ref-a"], detector_types=["D"],
                         min_impact=20000.0)

    sql = cur.sql[0].lower().replace(" ", "")
    assert "financial_impact_gbp,0)>=" in sql
    assert 20000.0 in cur.params[0]


def test_retire_keeps_findings_this_run_still_sees():
    cur = _Cur()
    store.retire_missing(cur, seen_ref_ids=["ref-a", "ref-b"],
                         detector_types=["D"])

    sql = cur.sql[0].lower()
    assert "not in" in sql or "<> all" in sql or "not = any" in sql
    assert any("ref-a" in str(p) for p in cur.params[0])


# ---------------------------------------------------------------------------
# The miner only retires on a run that actually looked at everything
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_data,expected", [
    ({"workflow": "all", "conditions": {}}, True),
    ({"workflow": "all", "conditions": {}, "supplier_id": "SUP-1"}, False),
    ({"workflow": "all", "conditions": {"supplier_id": "SUP-1"}}, False),
    ({"workflow": "price_benchmark", "conditions": {}}, False),
    ({"conditions": {}}, False),
])
def test_only_an_unscoped_run_may_retire(input_data, expected):
    """A filtered run must never retire findings it was never asked to look at."""
    from agents.opportunity_miner_agent import _run_covers_whole_corpus

    assert _run_covers_whole_corpus(input_data) is expected
