from scripts.testdata.verify import (
    CHECKS,
    CheckResult,
    blocking_failures,
)


def test_fourteen_checks_with_twelve_blocking():
    assert len(CHECKS) == 14
    assert sum(1 for check in CHECKS if check.blocking) == 12


def test_check_refs_are_sequential_and_unique():
    refs = [check.ref for check in CHECKS]
    assert refs == [f"V{i:02d}" for i in range(1, 15)]


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
