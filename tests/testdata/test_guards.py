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


def test_ingested_data_error_is_distinct_from_an_unsafe_target():
    from scripts.testdata.guards import IngestedDataError, UnsafeTargetError

    assert not issubclass(IngestedDataError, UnsafeTargetError)
    assert issubclass(IngestedDataError, RuntimeError)


def test_assert_no_ingested_documents_passes_on_a_purely_seeded_database(monkeypatch):
    from scripts.testdata import guards

    monkeypatch.setattr(guards, "count_ingested_documents", lambda db: {})
    assert guards.assert_no_ingested_documents("bp_testdb") is None


def test_assert_no_ingested_documents_names_the_tables_and_counts(monkeypatch):
    from scripts.testdata import guards

    monkeypatch.setattr(
        guards, "count_ingested_documents",
        lambda db: {"bp_invoice_trgt": 3, "bp_quote_trgt": 1},
    )
    with pytest.raises(guards.IngestedDataError) as excinfo:
        guards.assert_no_ingested_documents("bp_testdb")

    message = str(excinfo.value)
    assert "bp_invoice_trgt: 3" in message
    assert "bp_quote_trgt: 1" in message
    assert "--force" in message
