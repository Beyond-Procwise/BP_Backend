import pytest

from scripts.testdata.db import copy_rows, create_database
from scripts.testdata.guards import UnsafeTargetError


@pytest.mark.parametrize("name", ["bp_sqldb", "uicanvas"])
def test_create_database_refuses_live_targets(name):
    with pytest.raises(UnsafeTargetError):
        create_database(name)


@pytest.mark.parametrize("name", ["bp_sqldb", "uicanvas"])
def test_drop_first_also_refuses_live_targets(name):
    with pytest.raises(UnsafeTargetError):
        create_database(name, drop_first=True)


@pytest.mark.integration
def test_copy_rows_inserts_and_returns_count():
    from scripts.testdata.db import connect

    conn = connect("bp_testdb")
    try:
        with conn.cursor() as cur:
            cur.execute("create schema if not exists scratch")
            cur.execute("drop table if exists scratch.copy_probe")
            cur.execute("create table scratch.copy_probe (a text, b int)")
        conn.commit()

        written = copy_rows(
            conn, "scratch", "copy_probe", ["a", "b"], [("x", 1), ("y", 2)]
        )
        conn.commit()
        assert written == 2

        with conn.cursor() as cur:
            cur.execute("select a, b from scratch.copy_probe order by a")
            assert cur.fetchall() == [("x", 1), ("y", 2)]
            cur.execute("drop table scratch.copy_probe")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.integration
def test_copy_rows_writes_null_for_none():
    from scripts.testdata.db import connect

    conn = connect("bp_testdb")
    try:
        with conn.cursor() as cur:
            cur.execute("create schema if not exists scratch")
            cur.execute("drop table if exists scratch.null_probe")
            cur.execute("create table scratch.null_probe (a text, b numeric)")
        conn.commit()

        copy_rows(conn, "scratch", "null_probe", ["a", "b"], [("only", None)])
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("select a, b from scratch.null_probe")
            assert cur.fetchall() == [("only", None)]
            cur.execute("drop table scratch.null_probe")
        conn.commit()
    finally:
        conn.close()
