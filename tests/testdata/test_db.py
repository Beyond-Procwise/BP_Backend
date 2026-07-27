import pytest

from scripts.testdata.db import copy_rows, create_database
from scripts.testdata.guards import UnsafeTargetError
from tests.testdata import SCRATCH_DB


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

    conn = connect(SCRATCH_DB)
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
def test_copy_rows_round_trips_jsonb_verbatim():
    """psycopg2 hands back jsonb as a Python dict; str(dict) is not JSON."""
    from scripts.testdata.db import connect

    conn = connect(SCRATCH_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("create schema if not exists scratch")
            cur.execute("drop table if exists scratch.json_probe")
            cur.execute("create table scratch.json_probe (a text, b jsonb)")
            cur.execute(
                """insert into scratch.json_probe values
                   ('src', '{"rules": {"weights": {"risk": 0.2}}, "note": "it''s fine"}')"""
            )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("select a, b from scratch.json_probe")
            source_rows = cur.fetchall()

        copy_rows(
            conn, "scratch", "json_probe", ["a", "b"],
            [("copy", source_rows[0][1])],
        )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("select b from scratch.json_probe where a = 'src'")
            original = cur.fetchone()[0]
            cur.execute("select b from scratch.json_probe where a = 'copy'")
            copied = cur.fetchone()[0]
            assert copied == original
            cur.execute("drop table scratch.json_probe")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.integration
def test_copy_rows_round_trips_an_integer_array():
    """psycopg2 hands back int[] as a Python list; str(list) is not an array literal."""
    from scripts.testdata.db import connect

    conn = connect(SCRATCH_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("create schema if not exists scratch")
            cur.execute("drop table if exists scratch.array_probe")
            cur.execute("create table scratch.array_probe (a text, b int[], c text[])")
        conn.commit()

        copy_rows(
            conn, "scratch", "array_probe", ["a", "b", "c"],
            [("one", [1, 2, 3], ["plain", 'has,comma', 'has"quote']), ("empty", [], None)],
        )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("select a, b, c from scratch.array_probe order by a")
            assert cur.fetchall() == [
                ("empty", [], None),
                ("one", [1, 2, 3], ["plain", "has,comma", 'has"quote']),
            ]
            cur.execute("drop table scratch.array_probe")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.integration
def test_copy_rows_writes_null_for_none():
    from scripts.testdata.db import connect

    conn = connect(SCRATCH_DB)
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
