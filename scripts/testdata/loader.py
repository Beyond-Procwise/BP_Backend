"""Bulk-load mapped rows into a target database.

Shared by every stage. The mapping modules (persist.py, persist_org.py) are pure
and produce `{table: [row, ...]}` in COLUMNS order; this is the only place that
touches a database, so the safety guard and the required-set check sit in one
place rather than being repeated per stage.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

from scripts.testdata.db import connect, copy_rows
from scripts.testdata.guards import assert_safe_target


class MissingRequiredValue(RuntimeError):
    """A column declared required came out empty."""


def _check_required(
    table: str,
    columns: Sequence[str],
    required: Sequence[str],
    rows: Sequence[Sequence],
) -> None:
    """These tables carry almost no NOT NULL constraint, so the database will
    accept a row of nulls without complaint. This is what catches it."""
    positions = [(list(columns).index(name), name) for name in required]
    for number, row in enumerate(rows, start=1):
        for position, name in positions:
            if row[position] is None or row[position] == "":
                raise MissingRequiredValue(
                    f"{table}.{name} is required but empty (row {number})"
                )


def load_tables(
    target_db: str,
    columns: Mapping[str, Sequence[str]],
    required: Mapping[str, Sequence[str]],
    rows: Mapping[str, Sequence[Sequence]],
    *,
    check: Callable[[str, Sequence[Sequence]], None] | None = None,
    order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Truncate and bulk-load each table. Returns rows written per table.

    Every row is validated against the table's required set *before* any
    database contact, so a mapping mistake fails without leaving a half-loaded
    database behind.

    `order` controls load sequence where foreign keys matter -- cost centres
    reference business units, so the units must land first. Defaults to the
    order the rows mapping presents.
    """
    assert_safe_target(target_db)

    tables = list(order) if order is not None else list(rows)
    missing = [table for table in tables if table not in rows]
    if missing:
        raise KeyError(f"no rows supplied for {missing}")

    for table in tables:
        if check is not None:
            check(table, rows[table])
        else:
            _check_required(table, columns[table], required.get(table, ()), rows[table])

    written: dict[str, int] = {}
    conn = connect(target_db)
    try:
        # One statement for the whole group. The line tables carry foreign keys
        # onto their headers, and Postgres refuses to truncate a referenced
        # table on its own -- so emptying them one at a time cannot work.
        group = ", ".join(f'proc."{table}"' for table in tables)
        with conn.cursor() as cur:
            cur.execute(f"truncate {group}")

        for table in tables:
            written[table] = copy_rows(
                conn, "proc", table, list(columns[table]), rows[table]
            )
        conn.commit()
    finally:
        conn.close()
    return written
