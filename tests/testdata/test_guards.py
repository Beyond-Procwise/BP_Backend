import pytest

from scripts.testdata.guards import (
    UnsafeTargetError,
    assert_live_unchanged,
    assert_safe_target,
)


@pytest.mark.parametrize("name", ["bp_sqldb", "uicanvas", "ses", "postgres", "rdsadmin"])
def test_live_database_names_are_refused(name):
    with pytest.raises(UnsafeTargetError, match="refuses"):
        assert_safe_target(name)


@pytest.mark.parametrize("name", ["BP_SQLDB", "  uicanvas  ", "UiCanvas"])
def test_refusal_ignores_case_and_whitespace(name):
    with pytest.raises(UnsafeTargetError):
        assert_safe_target(name)


@pytest.mark.parametrize("name", ["bp_testdb", "uicanvas_test", "scratch_db"])
def test_test_database_names_are_allowed(name):
    assert assert_safe_target(name) is None


@pytest.mark.parametrize("name", ["", "   ", None])
def test_empty_target_is_refused(name):
    with pytest.raises(UnsafeTargetError):
        assert_safe_target(name)


def test_unchanged_live_counts_pass():
    counts = {"bp_sqldb.proc.bp_supplier": 123, "uicanvas.proc.supplier": 1009}
    assert assert_live_unchanged(counts, dict(counts)) is None


def test_changed_live_counts_raise_and_name_the_table():
    before = {"bp_sqldb.proc.bp_supplier": 123, "uicanvas.proc.supplier": 1009}
    after = {"bp_sqldb.proc.bp_supplier": 5123, "uicanvas.proc.supplier": 1009}
    with pytest.raises(UnsafeTargetError, match="bp_sqldb.proc.bp_supplier"):
        assert_live_unchanged(before, after)


def test_disappearing_table_raises():
    with pytest.raises(UnsafeTargetError, match="proc.gone"):
        assert_live_unchanged({"bp_sqldb.proc.gone": 5}, {})
