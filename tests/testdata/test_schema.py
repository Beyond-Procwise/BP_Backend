import pytest

from scripts.testdata.guards import UnsafeTargetError
from scripts.testdata.schema import SCHEMA_PAIRS, clone_schema


def test_schema_pairs_map_live_to_test_databases():
    assert SCHEMA_PAIRS == (
        ("bp_sqldb", "bp_testdb"),
        ("uicanvas", "uicanvas_test"),
    )


@pytest.mark.parametrize("target", ["bp_sqldb", "uicanvas"])
def test_clone_refuses_to_write_into_a_live_database(target):
    with pytest.raises(UnsafeTargetError):
        clone_schema("bp_sqldb", target)


@pytest.mark.integration
def test_clone_reproduces_tables_and_views_but_no_rows():
    from scripts.testdata.db import connect

    report = clone_schema("bp_sqldb", "bp_testdb", drop_first=True)
    assert report.tables >= 100
    assert report.views >= 5

    conn = connect("bp_testdb")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from information_schema.tables "
                "where table_schema = 'proc' and table_type = 'BASE TABLE'"
            )
            assert cur.fetchone()[0] >= 100

            cur.execute("select count(*) from proc.bp_supplier")
            assert cur.fetchone()[0] == 0

            cur.execute(
                "select count(*) from information_schema.views where table_schema = 'proc'"
            )
            assert cur.fetchone()[0] >= 5
    finally:
        conn.close()
