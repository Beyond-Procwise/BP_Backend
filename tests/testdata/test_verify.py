import pytest

from scripts.testdata import verify
from scripts.testdata.verify import (
    CHECKS,
    CheckResult,
    blocking_failures,
)


def test_new_checks_are_registered_and_no_longer_skip():
    refs = {c.ref for c in verify.CHECKS}
    assert {"V01", "V04", "V05"} <= refs
    assert "V15" in refs, "the benchmark check needs its own ref"


def test_benchmark_check_is_blocking():
    assert verify.CHECK_BY_REF["V15"].blocking


def test_fourteen_checks_with_twelve_blocking():
    # V15 (benchmark computes) landed after this suite's original 14/12 count.
    assert len(CHECKS) == 15
    assert sum(1 for check in CHECKS if check.blocking) == 13


def test_check_refs_are_sequential_and_unique():
    refs = [check.ref for check in CHECKS]
    assert refs == [f"V{i:02d}" for i in range(1, 16)]


def test_scored_checks_are_the_two_answer_key_scores():
    scored = [check.ref for check in CHECKS if not check.blocking]
    assert scored == ["V10", "V12"]


def test_blocking_failures_ignores_non_blocking_checks():
    results = [
        CheckResult(ref="V10", passed=False, detail="82% of defects found"),
        CheckResult(ref="V12", passed=False, detail="94% extraction accuracy"),
    ]
    assert blocking_failures(results) == []


def test_blocking_failures_reports_blocking_checks():
    results = [
        CheckResult(ref="V01", passed=False, detail="bp_supplier has 4,998 rows"),
        CheckResult(ref="V02", passed=True, detail="no orphans"),
    ]
    failures = blocking_failures(results)
    assert [failure.ref for failure in failures] == ["V01"]


def test_blocking_failures_passes_a_clean_run():
    results = [CheckResult(ref=check.ref, passed=True, detail="ok") for check in CHECKS]
    assert blocking_failures(results) == []


def test_v07_is_declared_blocking():
    from scripts.testdata.verify import CHECK_BY_REF

    assert CHECK_BY_REF["V07"].blocking is True


@pytest.mark.integration
def test_rollup_check_passes_on_a_loaded_scratch_database(scratch_uicanvas_schema):
    from scripts.testdata.catalogue import build_catalogue
    from scripts.testdata.loader import load_tables
    from scripts.testdata.org import build_business_units, build_cost_centres
    from scripts.testdata.persist_org import COLUMNS, REQUIRED, rows_for_org
    from scripts.testdata.reference import TaxonomyLeaf
    from scripts.testdata.verify import check_rollup
    from tests.testdata import SCRATCH_UICANVAS_DB

    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2="Software", l3="ERP", l4=f"Sub{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000",
            l4_id=f"C-45{i:02d}", l5_id=f"C-51{i:02d}",
            unspsc_code=str(10000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(246)
    ]
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(500)])
    load_tables(
        SCRATCH_UICANVAS_DB, COLUMNS, REQUIRED, rows_for_org(units, centres, items),
        order=("business_unit", "cost_centre", "item"),
    )

    result = check_rollup(SCRATCH_UICANVAS_DB)
    assert result.ref == "V07"
    assert result.passed, result.detail
    assert "entity" in result.detail.lower()


@pytest.mark.integration
def test_rollup_check_fails_when_a_cost_centre_points_nowhere(scratch_uicanvas_schema):
    from scripts.testdata.db import connect
    from scripts.testdata.verify import check_rollup
    from tests.testdata import SCRATCH_UICANVAS_DB

    conn = connect(SCRATCH_UICANVAS_DB)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "insert into proc.cost_centre (cost_centre_level_id, business_unit_id) "
                "values ('CC-ORPHAN', 'BU-DOES-NOT-EXIST')"
            )
        conn.commit()
        result = check_rollup(SCRATCH_UICANVAS_DB)
        assert not result.passed
        assert "1 unresolved" in result.detail
    finally:
        with conn.cursor() as cur:
            cur.execute("delete from proc.cost_centre where cost_centre_level_id = 'CC-ORPHAN'")
        conn.commit()
        conn.close()


# --- V14: isolation under concurrent live activity ---------------------------

def test_v14_fails_when_seeder_rows_turn_up_in_live(monkeypatch):
    from scripts.testdata import verify

    monkeypatch.setattr(
        verify, "find_seeded_rows_in_live",
        lambda *a, **k: {"bp_sqldb.proc.bp_invoice_trgt": 12},
    )
    monkeypatch.setattr(verify, "snapshot_counts", lambda dbs: {})
    result = verify.check_live_untouched({})
    assert not result.passed
    assert "bp_invoice_trgt (12)" in result.detail


def test_v14_fails_when_a_table_the_seeder_writes_moves(monkeypatch):
    from scripts.testdata import verify

    monkeypatch.setattr(verify, "find_seeded_rows_in_live", lambda *a, **k: {})
    monkeypatch.setattr(
        verify, "snapshot_counts",
        lambda dbs: {"bp_sqldb.proc.bp_invoice_trgt": 51},
    )
    result = verify.check_live_untouched(
        {"bp_sqldb.proc.bp_invoice_trgt": 50}, seeded_tables=["bp_invoice_trgt"],
    )
    assert not result.passed
    assert "bp_invoice_trgt" in result.detail


def test_v14_tolerates_the_production_service_doing_its_own_work(monkeypatch):
    """The audit log grows while the build runs. That is not a breach, and a
    check that calls it one cannot ever pass on a live cluster."""
    from scripts.testdata import verify

    monkeypatch.setattr(verify, "find_seeded_rows_in_live", lambda *a, **k: {})
    monkeypatch.setattr(
        verify, "snapshot_counts",
        lambda dbs: {"bp_sqldb.proc.bp_agent_actions": 54637},
    )
    result = verify.check_live_untouched(
        {"bp_sqldb.proc.bp_agent_actions": 54152}, seeded_tables=["bp_invoice_trgt"],
    )
    assert result.passed, result.detail
    assert "concurrent service activity" in result.detail
    assert "bp_agent_actions" in result.detail


def test_v14_passes_cleanly_when_nothing_moved(monkeypatch):
    from scripts.testdata import verify

    monkeypatch.setattr(verify, "find_seeded_rows_in_live", lambda *a, **k: {})
    monkeypatch.setattr(verify, "snapshot_counts", lambda dbs: {"a": 1})
    result = verify.check_live_untouched({"a": 1})
    assert result.passed
    assert "no seeder rows in live" in result.detail
    assert "concurrent" not in result.detail
